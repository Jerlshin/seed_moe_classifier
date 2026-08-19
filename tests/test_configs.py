"""Hydra composition and agreement with the paper's stated hyperparameters.

Every assertion here cites the paper. A failure means the configs have drifted
from Table 1 / Section 5, not that a test is stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from tests.conftest import (
    ADACOS_SCALE_27,
    DATASET_NUM_CROPS,
    FIRST_REVISION_DINO_OUT_DIM,
    LR_BASE,
    LR_REFERENCE_BATCH,
    REVISED_BACKBONE_FEATURE_DIM,
    SHIPPED_BACKBONE,
    SHIPPED_BACKBONE_FEATURE_DIM,
    REVISED_DINO_BOTTLENECK_DIM,
    REVISED_DINO_HEAD_LAYERS,
    REVISED_DINO_HIDDEN_DIM,
    REVISED_DROP_PATH_RATE,
    REVISED_EFFECTIVE_BATCH,
    REVISED_EPOCHS,
    REVISED_LEARNING_RATE,
    REVISED_LR_WARMUP_EPOCHS,
    REVISED_PHYSICAL_BATCH,
    SUBMITTED_BACKBONE_FEATURE_DIM,
    SUBMITTED_EPOCHS,
    REVISED_CENTER_MOMENTUM,
    REVISED_DINO_OUT_DIM,
    REVISED_TEACHER_MOMENTUM_FINAL,
    REVISED_TEACHER_TEMP,
    REVISED_WARMUP_EPOCHS,
    REVISED_WARMUP_TEACHER_TEMP,
    SUBMITTED_ARCFACE_SCALE,
    SUBMITTED_CENTER_MOMENTUM,
    PAPER_CLIP_GRAD,
    SUBMITTED_DINO_OUT_DIM,
    PAPER_EMBED_DIM,
    PAPER_LOCAL_CROPS,
    PAPER_NUM_EXPERTS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    SUBMITTED_TEACHER_MOMENTUM,
    SUBMITTED_TEACHER_TEMP,
    SUBMITTED_WARMUP_EPOCHS,
    SUBMITTED_WARMUP_TEACHER_TEMP,
    REVISED_TOP_K,
    SUBMITTED_TOP_K,
)


def build(conf_dir: str, *overrides: str):
    """Compose the config the way a real run does.

    ``return_hydra_config=True`` plus ``HydraConfig.set_config`` is what makes
    ``${hydra:runtime.output_dir}`` resolvable outside ``@hydra.main``; without
    it every tracking path would raise on resolution.

    The ``hydra`` node itself is then dropped: Hydra marks it read-only, so
    leaving it in place would make ``OmegaConf.resolve`` fail on Hydra's own
    internal interpolations rather than on anything this project owns.
    """
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        composed = compose(config_name="config", overrides=list(overrides), return_hydra_config=True)
        HydraConfig.instance().set_config(composed)

    cfg = OmegaConf.create(OmegaConf.to_container(composed, resolve=False))
    cfg.pop("hydra", None)
    return cfg


@pytest.fixture
def pretrain_cfg(conf_dir):
    return build(conf_dir, "experiment=pretrain_swinv2_dino")


@pytest.fixture
def finetune_cfg(conf_dir):
    return build(conf_dir, "experiment=finetune_hierarchical_moe")


# ------------------------------------------------------------- composition


#: Every experiment file. A new one that is not listed here is a config that
#: nothing composes until someone launches a suite against it, which is the
#: worst moment to discover a dangling interpolation.
ALL_EXPERIMENTS = [
    "pretrain_swinv2_dino",
    "pretrain_swinv2_base_dino",
    # Stage-1 trunk options. `screen_backbones` decides between them on measured
    # transfer rather than on capacity; until it has run, neither is adopted.
    "pretrain_swinv2_tiny_dino",
    "pretrain_swinv2_base_in22k_dino",
    "eval_pretrain_representation",
    # Phase-0 screening and the frozen-trunk reference. Neither trains anything;
    # both must still compose, because the failure mode of a dangling
    # interpolation is discovering it when a suite launches.
    "screen_backbones",
    "eval_frozen_reference",
    "control_imagenet_frozen",
    "finetune_hierarchical_moe",
    "ablation_flat_classifier",
    "baseline_resnet50",
    "baseline_swin_tiny",
    "baseline_hierarchical_cce",
    "baseline_linear_probe",
    "baseline_swinv2_supervised",
]


def test_all_experiment_files_are_covered(conf_dir):
    """The list above must not drift behind `conf/experiment/`."""
    on_disk = sorted(path.stem for path in (Path(conf_dir) / "experiment").glob("*.yaml"))
    assert on_disk == sorted(ALL_EXPERIMENTS)


@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_every_experiment_composes_and_resolves(conf_dir, experiment):
    cfg = build(conf_dir, f"experiment={experiment}")
    OmegaConf.resolve(cfg)  # raises on a dangling interpolation
    assert cfg.experiment.name == experiment


def test_experiments_select_the_right_head_and_loss(pretrain_cfg, finetune_cfg):
    """conf/model/* is genuinely composed, not shadowed by inline duplicates."""
    assert pretrain_cfg.model.head.name == "dino_projection"
    # Renamed: stage 1 is DINO self-distillation with two DINOv2 components, not
    # DINOv2. Calling it "dino" invited the reading the paper could not support.
    assert pretrain_cfg.model.loss.name == "dino_self_distillation"
    assert finetune_cfg.model.head.name == "hierarchical_moe"
    assert finetune_cfg.model.loss.name == "combined_arcface_kl_moe"


def test_only_swinv2_is_available_as_a_dinov2_backbone(conf_dir):
    """The comparative ViT-S/14 path was removed; SwinV2 must be the only option."""
    backbones = sorted(path.stem for path in (Path(conf_dir) / "model" / "backbone").glob("*.yaml"))
    assert backbones == ["swinv2"]

    cfg = build(conf_dir, "experiment=finetune_hierarchical_moe")
    assert cfg.model.backbone.name.startswith("swinv2")


def test_encoder_projects_to_the_paper_embedding_dimension(finetune_cfg):
    """Whatever SwinV2 emits natively, the head still receives z in R^384."""
    # SwinV2-Tiny's own width. The projection to z is what makes the Tiny/Base
    # swap invisible to the head.
    assert finetune_cfg.model.backbone.feature_dim == REVISED_BACKBONE_FEATURE_DIM
    assert finetune_cfg.model.head.embed_dim == PAPER_EMBED_DIM
    assert finetune_cfg.model.head.feature_dim == PAPER_EMBED_DIM


def test_command_line_overrides_reach_the_loss(conf_dir):
    cfg = build(conf_dir, "experiment=finetune_hierarchical_moe", "model.loss.lambda_kl=0.25")
    assert cfg.model.loss.lambda_kl == pytest.approx(0.25)


# ----------------------------------------------------------- paper: Table 1


def test_dino_pretraining_keeps_the_table_1_values_it_should(pretrain_cfg):
    """The Table 1 entries the revision did not change."""
    training = pretrain_cfg.experiment.training

    assert pretrain_cfg.model.backbone.name.startswith("swinv2")   # "Swin Transformer v2"
    assert training.clip_grad == pytest.approx(PAPER_CLIP_GRAD)     # "Clip Gradient 3"
    # Table 1's 0.996 is now the *start* of a cosine schedule, not a constant.
    assert training.momentum_teacher == pytest.approx(SUBMITTED_TEACHER_MOMENTUM)
    # Multi-crop is unchanged and deliberately so: whether the four local crops
    # earn their cost is an ablation to run, not an assumption to act on.
    assert pretrain_cfg.data.augmentation.local_crops_number == PAPER_LOCAL_CROPS


def test_stage_one_starts_from_imagenet_and_trains_the_trunk(pretrain_cfg):
    """ImageNet -> SwinV2-Small -> DINO fine-tuning, not ImageNet feature extraction.

    ``freeze=false`` is the assertion that matters. Stage 1 with a frozen trunk
    fits the projection head to fixed features and publishes an unadapted
    encoder, and nothing in the loss curve would say so -- which is why
    ``build_dino`` refuses it outright rather than trusting this config.

    The trunk is ``SHIPPED_BACKBONE`` (Small), not ``REVISED_BACKBONE`` (Tiny):
    the config moved to Small and the published 100-epoch checkpoint is a Small.
    Asserting the Tiny name here would make the *test* the thing that is wrong.
    """
    backbone = pretrain_cfg.model.backbone
    assert backbone.name == SHIPPED_BACKBONE
    assert backbone.feature_dim == SHIPPED_BACKBONE_FEATURE_DIM
    assert backbone.pretrained is True
    assert backbone.freeze is False
    assert backbone.drop_path_rate == pytest.approx(REVISED_DROP_PATH_RATE)


def test_stage_two_backbone_defaults_keep_the_checkpoint_path(finetune_cfg):
    """The ImageNet flag belongs to the stage-1 experiment, not to the group.

    A global ``pretrained: true`` would make every stage-2 run download ImageNet
    weights and then overwrite them with the stage-1 checkpoint -- wasted on a
    server and fatal on a machine with no network access.
    """
    backbone = finetune_cfg.model.backbone
    assert backbone.pretrained is False
    assert backbone.freeze is True
    assert "dinov2_swinv2_pretrained.pth" in str(backbone.checkpoint_path)


def test_frozen_imagenet_control_is_the_lower_arm_of_the_stage_one_comparison(conf_dir):
    """A = ImageNet frozen; B = ImageNet + DINO. B - A is what stage 1 contributes."""
    cfg = build(conf_dir, "experiment=control_imagenet_frozen")
    OmegaConf.resolve(cfg)
    backbone = cfg.model.backbone
    assert backbone.pretrained is True
    assert backbone.freeze is True
    # Null, or it would silently load the self-supervised encoder and stop being
    # a control.
    assert backbone.checkpoint_path is None
    # Everything else must match the run it is compared against.
    reference = build(conf_dir, "experiment=finetune_hierarchical_moe")
    OmegaConf.resolve(reference)
    assert cfg.model.head == reference.model.head
    assert cfg.model.loss == reference.model.loss
    assert cfg.experiment.training.epochs == reference.experiment.training.epochs
    assert cfg.experiment.training.split_protocol == reference.experiment.training.split_protocol


def test_base_capacity_control_changes_only_the_trunk(conf_dir):
    """SwinV2-Base at the identical recipe, publishing to its own path."""
    cfg = build(conf_dir, "experiment=pretrain_swinv2_base_dino")
    reference = build(conf_dir, "experiment=pretrain_swinv2_dino")
    OmegaConf.resolve(cfg)
    OmegaConf.resolve(reference)

    assert cfg.model.backbone.name == "swinv2_base_window16_256"
    assert cfg.model.backbone.feature_dim == SUBMITTED_BACKBONE_FEATURE_DIM
    # The head's in_dim follows feature_dim, so the two cannot disagree.
    assert cfg.model.head.in_dim == SUBMITTED_BACKBONE_FEATURE_DIM
    # Identical recipe otherwise.
    for key in ("epochs", "warmup_epochs", "lr_base", "effective_batch_size", "save_epochs"):
        assert cfg.experiment.training[key] == reference.experiment.training[key]
    assert cfg.data.batch_size == reference.data.batch_size
    # And a DIFFERENT publication path: writing the shared one would swap the
    # trunk under every stage-2 variant with only a "missing keys" log line.
    assert (
        cfg.experiment.training.shared_backbone_path
        != reference.experiment.training.shared_backbone_path
    )


def test_stage_one_duration_and_milestones(pretrain_cfg):
    """100 epochs, with 25/50/100 kept permanently so the length can be ablated."""
    training = pretrain_cfg.experiment.training
    assert training.epochs == REVISED_EPOCHS
    assert training.epochs < SUBMITTED_EPOCHS
    assert list(training.save_epochs) == [25, 50, 100]
    assert max(training.save_epochs) == training.epochs
    # The rolling series must not fight the milestones for `keep_last_n`.
    assert training.save_interval == 0


def test_physical_batch_is_preferred_to_accumulation(pretrain_cfg):
    """Sinkhorn and KoLeo are per-micro-batch, so accumulation cannot stand in.

    The submitted 16x4 and this 32x1 have the same effective batch and are not
    the same run: the former gives four 16-sample assignments where the latter
    gives one 32-sample assignment.
    """
    training = pretrain_cfg.experiment.training
    assert pretrain_cfg.data.batch_size == REVISED_PHYSICAL_BATCH
    assert training.effective_batch_size == REVISED_EFFECTIVE_BATCH
    assert training.gradient_accumulation_steps == 1
    # Single GPU: effective batch / (micro x world) must be exactly 1.
    assert training.effective_batch_size == pretrain_cfg.data.batch_size


def test_learning_rate_is_derived_from_the_effective_batch(pretrain_cfg):
    """Section 6.1's 0.0005 is DINO's rate at batch 256, not at this batch.

    Quoting it verbatim next to batch 32 is an 8x overstatement of the step
    size, and a literal rate is also immune to every knob that moves the batch.
    The config therefore leaves `learning_rate` null and the trainer derives it.
    """
    from src.trainers.contrastive_pretrain import resolve_learning_rate
    import logging

    training = pretrain_cfg.experiment.training
    assert training.learning_rate is None, "a literal rate would not follow the batch"
    assert training.lr_base == pytest.approx(LR_BASE)
    assert training.lr_reference_batch_size == LR_REFERENCE_BATCH
    assert training.lr_scaling == "linear"

    resolved, provenance = resolve_learning_rate(
        pretrain_cfg, REVISED_EFFECTIVE_BATCH, logging.getLogger("test")
    )
    assert resolved == pytest.approx(REVISED_LEARNING_RATE)
    assert resolved == pytest.approx(6.25e-05)
    assert provenance["rule"] == "linear"

    # Doubling the batch doubles the rate; that is the whole point.
    doubled, _ = resolve_learning_rate(
        pretrain_cfg, 2 * REVISED_EFFECTIVE_BATCH, logging.getLogger("test")
    )
    assert doubled == pytest.approx(2 * resolved)


def test_learning_rate_warmup_is_configured(pretrain_cfg):
    """10 epochs of linear warmup; the submitted configuration had none."""
    training = pretrain_cfg.experiment.training
    assert training.warmup_epochs == REVISED_LR_WARMUP_EPOCHS
    assert training.warmup_epochs < training.epochs
    assert training.scheduler.name == "cosine"
    # null t_max means `epochs - warmup_epochs`, so the cosine finishes with the
    # run rather than being truncated by the warmup.
    assert training.scheduler.t_max is None


def test_dino_head_is_sized_for_the_tiny_trunk(pretrain_cfg):
    """768 -> 1024 -> 1024 -> 256 -> 2048, and none of it reaches stage 2."""
    head = pretrain_cfg.model.head
    assert head.in_dim == REVISED_BACKBONE_FEATURE_DIM
    assert head.hidden_dim == REVISED_DINO_HIDDEN_DIM
    assert head.bottleneck_dim == REVISED_DINO_BOTTLENECK_DIM
    assert head.num_layers == REVISED_DINO_HEAD_LAYERS
    assert head.out_dim == REVISED_DINO_OUT_DIM
    # The two DINO-specific behaviours are unchanged.
    assert head.use_batch_norm == "layer"
    assert head.norm_last_layer is True


def test_stage_one_collapse_guards_are_recalibrated(pretrain_cfg):
    """Sharpening and centering were both set toward collapse; neither is now.

    The submitted configuration ran a teacher at tau = 0.04 -- roughly twice as
    sharp as DINO's converged 0.07 -- while its counterweight, a 65,536-wide
    centering vector, was an EMA at m = 0.9 over 32 teacher vectors per step:
    ~320 effective samples for 65,536 dimensions, 1/64th the sample density DINO
    has. Sharpening was strengthened and centering weakened simultaneously.
    """
    loss = pretrain_cfg.model.loss

    assert loss.warmup_teacher_temp == pytest.approx(REVISED_WARMUP_TEACHER_TEMP)
    assert loss.teacher_temp == pytest.approx(REVISED_TEACHER_TEMP)
    assert loss.teacher_temp > SUBMITTED_TEACHER_TEMP, "the teacher must be softer, not sharper"
    assert loss.warmup_teacher_temp_epochs == REVISED_WARMUP_EPOCHS

    # Sinkhorn removes the dependence on a running mean entirely: the assignment
    # is normalised within the batch, so nothing is estimated across steps.
    assert loss.centering == "sinkhorn"
    assert loss.center_momentum == pytest.approx(REVISED_CENTER_MOMENTUM)
    assert loss.lambda_koleo > 0


def test_prototype_count_is_sized_for_this_dataset_and_batch(pretrain_cfg):
    """65,536 prototypes for 9,357 images is 7.00 per image; DINO's ratio is 0.051.

    That is 137x the prototype density DINO was tuned at. The second cut, from
    8,192 to 2,048, is about the BATCH rather than the dataset: Sinkhorn gives
    each prototype ``B_teacher / K`` of the assignment mass, so the prototype
    count and the physical batch are one decision. At 64 teacher views, 2,048
    prototypes carry 4x the evidence per column that 8,192 did.
    """
    out_dim = pretrain_cfg.model.head.out_dim
    assert out_dim == REVISED_DINO_OUT_DIM
    assert out_dim < FIRST_REVISION_DINO_OUT_DIM < SUBMITTED_DINO_OUT_DIM
    assert out_dim / DATASET_NUM_CROPS < 1.0, "fewer prototypes than training images"

    # Teacher views per Sinkhorn estimate: 2 global crops x the physical batch.
    teacher_views = 2 * pretrain_cfg.data.batch_size
    assert out_dim / teacher_views <= 32, (
        "each prototype column must carry a meaningful share of the batch's mass"
    )


def test_stage_one_schedules_the_momentum_and_weight_decay(pretrain_cfg):
    """DINO anneals both; the submitted configuration held both constant."""
    training = pretrain_cfg.experiment.training
    assert training.momentum_teacher_final == pytest.approx(REVISED_TEACHER_MOMENTUM_FINAL)
    assert training.weight_decay_final > training.weight_decay


def test_dino_optimizer_matches_section_6_1(pretrain_cfg):
    """"AdamW with an initial learning rate of 0.0005 ... cosine decay scheduler".

    The rate itself is checked by ``test_learning_rate_is_derived_from_the_
    effective_batch``: 0.0005 survives as ``lr_base``, the reference-batch value
    the scaling rule is applied to.
    """
    assert pretrain_cfg.experiment.training.optimizer.name == "AdamW"
    assert pretrain_cfg.experiment.training.lr_base == pytest.approx(0.0005)
    assert pretrain_cfg.experiment.training.scheduler.name == "cosine"
    assert pretrain_cfg.experiment.training.freeze_last_layer_epochs == 1


def test_dino_head_normalisation_decouples_teacher_from_student(pretrain_cfg):
    """Section 4 says batch norm; the EMA cannot carry it.

    ``update_momentum`` EMAs *parameters*, not *buffers*, so a BatchNorm head
    leaves teacher and student running statistics to diverge -- and the teacher
    sees 2 views per step against the student's 6, so their batch statistics
    differ even in principle. DINO's reference head uses no batch norm for
    transformer trunks for the same reason.
    """
    assert pretrain_cfg.model.head.use_batch_norm in {"layer", "none"}


def test_multi_crop_configuration_matches_table_1(pretrain_cfg):
    augmentation = pretrain_cfg.data.augmentation
    assert augmentation.local_crops_number == PAPER_LOCAL_CROPS
    assert list(augmentation.global_crops_scale) == [0.4, 1.0]
    assert list(augmentation.local_crops_scale) == [0.05, 0.4]


def test_augmentation_probabilities_match_section_6_1(pretrain_cfg):
    """Blur p=1.0 / p=0.1 / p=0.5 and solarization p=0.2."""
    augmentation = pretrain_cfg.data.augmentation
    assert augmentation.global_blur_prob_1 == pytest.approx(1.0)
    assert augmentation.global_blur_prob_2 == pytest.approx(0.1)
    assert augmentation.local_blur_prob == pytest.approx(0.5)
    assert augmentation.solarization_prob == pytest.approx(0.2)


def test_color_jitter_magnitudes_match_section_6_1(pretrain_cfg):
    """"brightness +/-0.4, contrast +/-0.4, saturation +/-0.2, hue +/-0.1"."""
    augmentation = pretrain_cfg.data.augmentation
    assert augmentation.color_jitter_brightness == pytest.approx(0.4)
    assert augmentation.color_jitter_contrast == pytest.approx(0.4)
    assert augmentation.color_jitter_saturation == pytest.approx(0.2)
    assert augmentation.color_jitter_hue == pytest.approx(0.1)


# ---------------------------------------------------------- paper: Section 5


def test_hierarchy_sizes_match_section_3(finetune_cfg):
    assert finetune_cfg.data.num_seed_types == PAPER_NUM_SEED_TYPES
    assert finetune_cfg.data.num_sub_varieties == PAPER_NUM_SUB_VARIETIES


def test_embedding_dimension_matches_equation_4(finetune_cfg):
    """"The DINOv2 encoder extracts a 384-dimensional feature vector"."""
    assert finetune_cfg.model.head.embed_dim == PAPER_EMBED_DIM


def test_moe_matches_the_revised_routing_width(finetune_cfg):
    """Section 5.2 specifies six experts; the revision routes the top 2 of them."""
    assert finetune_cfg.model.head.num_experts == PAPER_NUM_EXPERTS
    assert finetune_cfg.model.head.top_k == REVISED_TOP_K


def test_submitted_top_k_is_still_reachable_by_override(conf_dir):
    """The Top-4 configuration must remain reproducible for the paper's own numbers."""
    cfg = build(conf_dir, "experiment=finetune_hierarchical_moe", "model.head.top_k=4")
    assert cfg.model.head.top_k == SUBMITTED_TOP_K


def test_component_toggles_are_all_enabled_by_default(finetune_cfg):
    """The full model must have every block on; ablations turn them off explicitly."""
    head = finetune_cfg.model.head
    for flag in ("use_moe", "use_arcface", "use_residual", "use_cross_attention", "use_kl_loss"):
        assert head[flag] is True, f"{flag} should default to True"


def test_kl_toggle_propagates_from_the_head_to_the_loss(conf_dir):
    """One switch, two consumers: the loss must follow the head's `use_kl_loss`."""
    cfg = build(conf_dir, "experiment=finetune_hierarchical_moe", "model.head.use_kl_loss=false")
    OmegaConf.resolve(cfg)
    assert cfg.model.loss.use_kl_loss is False


@pytest.mark.parametrize(
    ("experiment", "expected"),
    [
        ("baseline_resnet50", "resnet50"),
        ("baseline_swin_tiny", "swin_tiny"),
    ],
)
def test_supervised_baselines_select_their_backbone(conf_dir, experiment, expected):
    cfg = build(conf_dir, f"experiment={experiment}")
    OmegaConf.resolve(cfg)
    assert cfg.model.head.name == "flat_supervised"
    assert cfg.model.head.baseline_model == expected
    assert cfg.model.loss.name == "flat_cce"
    # Same embedding width as the proposed model, so t-SNE panels stay comparable.
    assert cfg.model.head.embed_dim == PAPER_EMBED_DIM


def test_hierarchical_cce_baseline_removes_the_proposed_machinery(conf_dir):
    """The simple-hierarchy control keeps the coarse-to-fine link and nothing else."""
    cfg = build(conf_dir, "experiment=baseline_hierarchical_cce")
    OmegaConf.resolve(cfg)
    head = cfg.model.head
    assert head.name == "hierarchical_moe"     # same class, toggles flipped
    assert head.use_moe is False
    assert head.use_arcface is False
    assert head.use_cross_attention is False
    assert head.use_kl_loss is False
    assert head.use_residual is True           # without it there is no hierarchy left


def test_all_variants_share_one_pretrained_encoder(conf_dir):
    """Every DINOv2-path run must resolve to the same published checkpoint."""
    paths = {
        build(conf_dir, f"experiment={experiment}").model.backbone.checkpoint_path
        for experiment in ("finetune_hierarchical_moe", "ablation_flat_classifier",
                           "baseline_hierarchical_cce")
    }
    assert len(paths) == 1
    assert "dinov2_swinv2_pretrained.pth" in str(paths.pop())


def test_head_uses_the_paper_variants_by_default(finetune_cfg):
    assert finetune_cfg.model.head.seed_classifier_variant == "mlp"
    assert finetune_cfg.model.head.cross_attention_variant == "paper"


def test_every_paper_loss_component_is_weighted(finetune_cfg):
    """All six terms the paper describes must be active."""
    loss = finetune_cfg.model.loss
    assert loss.lambda_seed > 0        # Eq. 7
    assert loss.lambda_arcface > 0     # Eq. 13
    assert loss.lambda_kl > 0          # Eq. 10
    assert loss.lambda_moe_load > 0    # dispatch-aware load balancing
    assert loss.lambda_cosine > 0      # class compactness
    assert loss.lambda_moe_z > 0       # router z-loss
    assert loss.lambda_residual > 0    # Eq. 9 residual magnitude hinge

    # L1 sparsity is deliberately OFF. Under renormalize_top_k it has a null
    # space with respect to the module's output -- mass can move onto the
    # selected set without changing h -- so its only reliable effect is to cut
    # router entropy, which fights the load term. Kept as an ablation axis.
    assert loss.lambda_moe_sparsity == 0.0
    assert loss.moe_load_mode == "switch"


def test_arcface_scale_is_analytic_not_inherited_from_face_recognition(finetune_cfg):
    """Eq. 13 needs a margin and a scale, and the scale must suit C = 27.

    ``s = 30`` is ArcFace's value for 10^5-10^6 identities. AdaCos derives the
    fixed optimal scale as ``sqrt(2) log(C-1)``, which is 4.61 here -- so 30 was
    6.5x too large, put ``L_ArcFace`` at ~17.6 against ``L_seed = 1.386`` at
    initialisation, and saturated ``softmax(s cos)`` into a near-one-hot
    distribution that broke the KL term and any calibration analysis.
    """
    from src.models.components.arcface_head import adacos_scale, resolve_scale

    head = finetune_cfg.model.head
    assert head.arcface_margin > 0
    assert head.arcface_scale == "auto"

    resolved = resolve_scale(head.arcface_scale, PAPER_NUM_SUB_VARIETIES)
    assert resolved == pytest.approx(ADACOS_SCALE_27, abs=1e-3)
    assert resolved == pytest.approx(adacos_scale(PAPER_NUM_SUB_VARIETIES))
    assert resolved < SUBMITTED_ARCFACE_SCALE


def test_hierarchy_kl_is_decoupled_from_the_arcface_scale(finetune_cfg):
    """``lambda_kl`` and ``arcface_scale`` must not be one hyperparameter in disguise."""
    loss = finetune_cfg.model.loss
    assert loss.tau_kl > 0
    assert loss.kl_mode in {"forward", "jsd"}
    # The coarse head is already supervised by hard labels; letting L_KL push it
    # too lets the term be reduced by agreeing with a confidently wrong fine
    # prediction.
    assert loss.detach_kl_seed_target is True


def test_split_protocol_defaults_to_group_aware(finetune_cfg):
    """9,357 crops from 81 photographs: crop-level splitting is not a neutral choice."""
    assert finetune_cfg.experiment.training.split_protocol == "grouped"


def test_stage_two_augmentation_is_not_empty(finetune_cfg):
    """The submitted default saw each image once per epoch, deterministically."""
    training = finetune_cfg.experiment.training
    assert training.horizontal_flip_prob > 0
    assert training.random_resized_crop_scale is not None
    assert training.margin_warmup_fraction > 0


def test_ablation_disables_the_hierarchy(conf_dir):
    cfg = build(conf_dir, "experiment=ablation_flat_classifier")
    assert cfg.model.loss.lambda_seed == 0.0
    assert cfg.model.loss.lambda_kl == 0.0
    assert cfg.model.loss.lambda_arcface > 0
    assert cfg.model.head.use_moe is False


# ------------------------------------------------------------------ tracking


def test_dual_tracking_is_enabled_by_default(finetune_cfg):
    assert finetune_cfg.tracking.tensorboard.enabled is True
    assert finetune_cfg.tracking.wandb.enabled is True


def test_wandb_defaults_to_offline(finetune_cfg):
    """Runs must never block on network access or credentials."""
    assert finetune_cfg.tracking.wandb.mode == "offline"


def test_paper_result_artifacts_are_enabled(finetune_cfg):
    artifacts = finetune_cfg.tracking.artifacts
    assert artifacts.log_confusion_matrices is True
    assert artifacts.log_expert_utilization is True
    assert artifacts.log_tsne is True
    assert artifacts.log_per_class_tables is True
