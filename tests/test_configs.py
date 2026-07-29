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
    PAPER_CENTER_MOMENTUM,
    PAPER_CLIP_GRAD,
    PAPER_DINO_OUT_DIM,
    PAPER_EMBED_DIM,
    PAPER_LOCAL_CROPS,
    PAPER_NUM_EXPERTS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    PAPER_TEACHER_MOMENTUM,
    PAPER_TEACHER_TEMP,
    PAPER_WARMUP_EPOCHS,
    PAPER_WARMUP_TEACHER_TEMP,
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


ALL_EXPERIMENTS = [
    "pretrain_swinv2_dino",
    "finetune_hierarchical_moe",
    "ablation_flat_classifier",
    "baseline_resnet50",
    "baseline_swin_tiny",
    "baseline_hierarchical_cce",
]


@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_every_experiment_composes_and_resolves(conf_dir, experiment):
    cfg = build(conf_dir, f"experiment={experiment}")
    OmegaConf.resolve(cfg)  # raises on a dangling interpolation
    assert cfg.experiment.name == experiment


def test_experiments_select_the_right_head_and_loss(pretrain_cfg, finetune_cfg):
    """conf/model/* is genuinely composed, not shadowed by inline duplicates."""
    assert pretrain_cfg.model.head.name == "dino_projection"
    assert pretrain_cfg.model.loss.name == "dino"
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
    assert finetune_cfg.model.backbone.feature_dim == 1024   # SwinV2-Base's own width
    assert finetune_cfg.model.head.embed_dim == PAPER_EMBED_DIM
    assert finetune_cfg.model.head.feature_dim == PAPER_EMBED_DIM


def test_command_line_overrides_reach_the_loss(conf_dir):
    cfg = build(conf_dir, "experiment=finetune_hierarchical_moe", "model.loss.lambda_kl=0.25")
    assert cfg.model.loss.lambda_kl == pytest.approx(0.25)


# ----------------------------------------------------------- paper: Table 1


def test_dino_pretraining_matches_table_1(pretrain_cfg):
    training = pretrain_cfg.experiment.training
    loss = pretrain_cfg.model.loss

    assert pretrain_cfg.model.backbone.name.startswith("swinv2")   # "Swin Transformer v2"
    assert pretrain_cfg.data.batch_size == 16                       # "Batch Size 16"
    assert training.epochs == 300                                   # "Number of Epochs 300"
    assert training.clip_grad == pytest.approx(PAPER_CLIP_GRAD)     # "Clip Gradient 3"
    assert training.momentum_teacher == pytest.approx(PAPER_TEACHER_MOMENTUM)
    assert loss.warmup_teacher_temp == pytest.approx(PAPER_WARMUP_TEACHER_TEMP)
    assert loss.teacher_temp == pytest.approx(PAPER_TEACHER_TEMP)
    assert loss.warmup_teacher_temp_epochs == PAPER_WARMUP_EPOCHS
    assert pretrain_cfg.model.head.out_dim == PAPER_DINO_OUT_DIM    # "65,536"


def test_dino_optimizer_matches_section_6_1(pretrain_cfg):
    """"AdamW with an initial learning rate of 0.0005 ... cosine decay scheduler"."""
    assert pretrain_cfg.experiment.training.optimizer.name == "AdamW"
    assert pretrain_cfg.experiment.training.learning_rate == pytest.approx(0.0005)
    assert pretrain_cfg.experiment.training.scheduler.name == "cosine"
    assert pretrain_cfg.experiment.training.freeze_last_layer_epochs == 1


def test_dino_head_uses_batch_norm(pretrain_cfg):
    """Section 4: "an MLP with batch normalization and GELU activation functions"."""
    assert pretrain_cfg.model.head.use_batch_norm is True


def test_dino_centering_momentum(pretrain_cfg):
    """Eq. 3 with m = 0.9."""
    assert pretrain_cfg.model.loss.center_momentum == pytest.approx(PAPER_CENTER_MOMENTUM)


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
    assert loss.lambda_moe_load > 0    # entropy load balancing
    assert loss.lambda_moe_sparsity > 0  # L1 sparsity
    assert loss.lambda_cosine > 0      # residual compactness


def test_arcface_hyperparameters_are_present(finetune_cfg):
    """Eq. 13 needs both an angular margin m and a feature scale s."""
    assert finetune_cfg.model.head.arcface_margin > 0
    assert finetune_cfg.model.head.arcface_scale > 1


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
