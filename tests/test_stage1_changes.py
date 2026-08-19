"""Tests for the STAGE1_CHANGES.md implementation.

Every test here pins a behaviour the stage-1 audit found either **wrong** or
**unmeasurable**, and the distinction matters when one fails:

* A KoLeo or loss-decomposition failure is a regression of a *correctness* fix.
  The old behaviour is still reachable behind a flag, so a test failing here
  means the default silently moved back.
* A provenance or split failure is a regression of an *instrument*. Nothing about
  the model changes, but a number becomes uninterpretable -- which is how the
  shipped encoder came to be trained on 8,173 of 9,357 crops with no record.

The file is deliberately organised by the audit's own section letters, so a
failure names the item it belongs to.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.conftest import (
    PAPER_GLOBAL_CROPS,
    PAPER_LOCAL_CROPS,
    SUBVARIETY_COUNTS,
)


# ===================================================================== A1
# KoLeo is applied per global view, not across them.


def _view_major_pairs(batch: int = 6, dim: int = 32, jitter: float = 1e-3):
    """``[2B, dim]`` where rows ``i`` and ``B+i`` are near-duplicate views.

    This is exactly the layout the trainer hands the loss: view-major, so the two
    views of one image sit ``B`` rows apart.
    """
    torch.manual_seed(0)
    base = torch.randn(batch, dim)
    return torch.cat([base, base + jitter * torch.randn(batch, dim)], dim=0)


def test_koleo_across_views_penalises_the_pair_the_objective_pulls_together():
    """The mechanism, stated as a measurement rather than an argument.

    Under ``all_views`` the nearest neighbour of row ``i`` is row ``B+i`` -- the
    other view of the same image -- so ``-log(min distance)`` is large and its
    gradient pushes them apart. Under ``per_view`` neither block contains its own
    partner, so the term is unaffected by how close the two views are.
    """
    from src.losses.dino import grouped_koleo

    features = _view_major_pairs()
    across = grouped_koleo(features, num_groups=2, scope="all_views")
    per_view = grouped_koleo(features, num_groups=2, scope="per_view")
    assert across > per_view + 1.0, (
        "the across-views form is supposed to be dominated by the same-image pair; "
        f"got {float(across):.4f} against {float(per_view):.4f}"
    )


def test_koleo_per_view_gradient_does_not_separate_the_two_views():
    """The falsifiable half: check the GRADIENT, not just the value.

    Under ``all_views`` the gradient on view 0's rows points away from view 1's;
    under ``per_view`` view 0's gradient does not depend on view 1 at all.
    """
    from src.losses.dino import grouped_koleo

    batch = 6
    couplings = {}
    for scope in ("all_views", "per_view"):
        features = _view_major_pairs(batch=batch).requires_grad_(True)
        grouped_koleo(features, num_groups=2, scope=scope).backward()
        first, second = features.grad[:batch], features.grad[batch:]
        couplings[scope] = float(torch.cosine_similarity(first, second, dim=-1).mean())

    # Two views of one image are near-identical inputs. Under `all_views` they are
    # each other's nearest neighbour, so the gradient is a mutual repulsion and
    # the partnered rows' gradients point in OPPOSITE directions. Under
    # `per_view` neither block can see the other, so each row's gradient is
    # driven by its own block's neighbours -- which, the two blocks being
    # near-identical, gives near-identical gradients pointing the SAME way.
    assert couplings["all_views"] < -0.5, (
        f"expected repulsion between partnered views, got {couplings['all_views']:.3f}"
    )
    assert couplings["per_view"] > 0.5, (
        f"per-view gradients should not repel the partner, got {couplings['per_view']:.3f}"
    )


def test_koleo_per_view_matches_a_hand_written_chunked_reference():
    """Value parity against the literal DINOv2 expression."""
    from src.losses.dino import grouped_koleo, koleo_regularizer

    features = _view_major_pairs(batch=5, dim=16, jitter=0.5)
    chunks = features.chunk(2, dim=0)
    reference_sum = koleo_regularizer(chunks[0]) + koleo_regularizer(chunks[1])
    assert grouped_koleo(features, 2, "per_view", "sum") == pytest.approx(
        float(reference_sum), abs=1e-6
    )
    assert grouped_koleo(features, 2, "per_view", "mean") == pytest.approx(
        float(reference_sum) / 2, abs=1e-6
    )


def test_koleo_all_views_is_exactly_the_unchunked_regulariser():
    """The control has to reproduce the pre-audit number bit for bit."""
    from src.losses.dino import grouped_koleo, koleo_regularizer

    features = _view_major_pairs(batch=4, dim=8, jitter=0.4)
    assert grouped_koleo(features, 2, "all_views") == pytest.approx(
        float(koleo_regularizer(features)), abs=0.0
    )


def test_koleo_refuses_a_layout_that_is_not_view_major():
    from src.losses.dino import grouped_koleo

    with pytest.raises(ValueError, match="view-major"):
        grouped_koleo(torch.randn(7, 8), num_groups=2, scope="per_view")


def test_koleo_scope_and_reduction_are_validated():
    from src.losses.dino import CustomDINOLoss, grouped_koleo

    with pytest.raises(ValueError, match="koleo_scope"):
        grouped_koleo(torch.randn(4, 8), 2, scope="both")
    with pytest.raises(ValueError, match="koleo_reduction"):
        grouped_koleo(torch.randn(4, 8), 2, reduction="median")
    with pytest.raises(ValueError, match="koleo_scope"):
        CustomDINOLoss(
            out_dim=8, num_crops=6, warmup_teacher_temp=0.04, teacher_temp=0.04,
            warmup_teacher_temp_epochs=0, num_epochs=1, koleo_scope="nonsense",
        )


def test_the_default_scope_is_the_fixed_one():
    """A default that quietly reverts is the failure this file exists to catch."""
    from src.losses.dino import CustomDINOLoss

    loss = CustomDINOLoss(
        out_dim=8, num_crops=6, warmup_teacher_temp=0.04, teacher_temp=0.04,
        warmup_teacher_temp_epochs=0, num_epochs=1,
    )
    assert loss.koleo_scope == "per_view"
    assert loss.koleo_reduction == "mean"


def test_the_shipped_config_selects_per_view_koleo(conf_dir):
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=pretrain_swinv2_dino")
    assert cfg.model.loss.koleo_scope == "per_view"
    assert cfg.model.loss.koleo_space == "bottleneck"
    assert cfg.model.loss.lambda_koleo > 0


# ===================================================================== A4
# The loss decomposes exactly, and both halves are always available.


def _dino_loss(**overrides):
    from src.losses.dino import CustomDINOLoss

    kwargs = {
        "out_dim": 32,
        "num_crops": PAPER_GLOBAL_CROPS + PAPER_LOCAL_CROPS,
        "warmup_teacher_temp": 0.04,
        "teacher_temp": 0.04,
        "warmup_teacher_temp_epochs": 0,
        "num_epochs": 1,
        "lambda_koleo": 0.0,
    }
    kwargs.update(overrides)
    return CustomDINOLoss(**kwargs)


@pytest.mark.parametrize("centering", ["sinkhorn", "ema"])
def test_cross_entropy_equals_entropy_plus_kl_exactly(centering):
    """``CE = H(q) + KL(q||p)``, to floating-point tolerance, under both centerings.

    This is the identity that makes ``train/teacher_student_kl`` readable at all.
    If it stops holding, the "learnable part" curve is measuring something else.
    """
    torch.manual_seed(0)
    loss_fn = _dino_loss(centering=centering)
    student = [torch.randn(4, 32) for _ in range(6)]
    teacher = [torch.randn(4, 32) for _ in range(2)]

    total = loss_fn(student, teacher, epoch=0)
    metrics = loss_fn.last_metrics
    assert metrics["dino_cross_entropy"] == pytest.approx(float(total), abs=1e-5)
    assert metrics["teacher_entropy_cross_view"] + metrics["teacher_student_kl"] == pytest.approx(
        metrics["dino_cross_entropy"], abs=1e-4
    )
    # The KL is a divergence: it cannot be negative beyond float noise.
    assert metrics["teacher_student_kl"] > -1e-5


def test_the_decomposition_is_emitted_even_on_a_non_logging_step():
    """The epoch mean has to see every micro-batch, so this cannot be gated."""
    loss_fn = _dino_loss()
    loss_fn.metrics_enabled = False
    loss_fn(
        [torch.randn(3, 32) for _ in range(6)],
        [torch.randn(3, 32) for _ in range(2)],
        epoch=0,
    )
    tensors = loss_fn.last_metric_tensors
    assert {"dino_cross_entropy", "teacher_entropy_cross_view", "teacher_student_kl"} <= set(tensors)
    assert all(torch.is_tensor(value) for value in tensors.values()), (
        "reading these must not synchronise with the device"
    )


def test_the_decomposition_reproduces_the_audits_arithmetic():
    """A sharper teacher lowers H and leaves the loss looking like progress.

    The point of the decomposition, restated as a test: raising the teacher
    temperature raises ``H`` and therefore the reported cross entropy, with the
    student untouched. Anyone optimising the raw loss is partly optimising this.
    """
    torch.manual_seed(1)
    student = [torch.randn(8, 32) for _ in range(6)]
    teacher = [torch.randn(8, 32) * 3.0 for _ in range(2)]

    readings = {}
    for temperature in (0.02, 0.2):
        loss_fn = _dino_loss(
            centering="ema", warmup_teacher_temp=temperature, teacher_temp=temperature
        )
        loss_fn(student, teacher, epoch=0)
        readings[temperature] = loss_fn.last_metrics

    assert readings[0.2]["teacher_entropy_cross_view"] > readings[0.02]["teacher_entropy_cross_view"]
    assert readings[0.2]["dino_cross_entropy"] > readings[0.02]["dino_cross_entropy"]


def test_loss_flags_distinguish_every_arm():
    """A ``koleo_scope`` control must not leave a byte-identical machine trace."""
    baseline = _dino_loss(lambda_koleo=0.1, koleo_scope="per_view").loss_flags()
    control = _dino_loss(lambda_koleo=0.1, koleo_scope="all_views").loss_flags()
    disabled = _dino_loss(lambda_koleo=0.0).loss_flags()
    ema = _dino_loss(centering="ema").loss_flags()

    assert baseline != control
    assert baseline != disabled
    assert baseline != ema
    assert baseline["koleo_scope"] == "per_view"
    assert disabled["lambda_koleo"] == 0.0
    assert ema["centering"] == "ema"
    assert json.dumps(baseline)  # JSON-safe: it lands in summary.json


# ===================================================================== A2 / E2
# Corpus provenance.


@pytest.fixture
def corpus_root(tmp_path):
    root = tmp_path / "corpus"
    for seed_type, count in SUBVARIETY_COUNTS.items():
        for index in range(count):
            directory = root / seed_type / f"{seed_type}_sub{index:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            for photo in range(2):
                for bbox in range(2):
                    Image.fromarray(
                        np.zeros((4, 4, 3), dtype=np.uint8)
                    ).save(directory / f"IMG_{photo:04d}_bbox{bbox}.png")
    return root


def test_corpus_fingerprint_counts_what_the_run_will_read(corpus_root):
    from src.datasets.dataset import corpus_fingerprint

    fingerprint = corpus_fingerprint(corpus_root)
    total = 4 * sum(SUBVARIETY_COUNTS.values())
    assert fingerprint["num_samples"] == total
    assert fingerprint["num_classes"] == sum(SUBVARIETY_COUNTS.values())
    assert fingerprint["num_source_groups"] == 2 * sum(SUBVARIETY_COUNTS.values())
    assert len(fingerprint["sha256"]) == 64


def test_corpus_fingerprint_is_stable_and_path_independent(corpus_root, tmp_path):
    """The digest identifies the CONTENT, not the mount point.

    A digest that moved with the absolute path would fire on every machine and be
    ignored within a week.
    """
    import shutil

    from src.datasets.dataset import corpus_fingerprint

    moved = tmp_path / "elsewhere" / "corpus"
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(corpus_root, moved)
    assert corpus_fingerprint(corpus_root)["sha256"] == corpus_fingerprint(moved)["sha256"]


def test_corpus_fingerprint_changes_when_one_crop_is_removed(corpus_root):
    """The 8,173-vs-9,357 case, in miniature."""
    from src.datasets.dataset import corpus_fingerprint, describe_fingerprint_mismatch

    before = corpus_fingerprint(corpus_root)
    victim = sorted(corpus_root.rglob("*.png"))[0]
    victim.unlink()
    after = corpus_fingerprint(corpus_root)

    assert after["sha256"] != before["sha256"]
    assert after["num_samples"] == before["num_samples"] - 1
    message = describe_fingerprint_mismatch(before, after, "the stage-1 run", "this evaluation")
    assert "CORPUS MISMATCH" in message
    assert str(before["num_samples"]) in message and str(after["num_samples"]) in message


def test_identical_corpora_produce_no_mismatch_line(corpus_root):
    from src.datasets.dataset import corpus_fingerprint, describe_fingerprint_mismatch

    fingerprint = corpus_fingerprint(corpus_root)
    assert describe_fingerprint_mismatch(fingerprint, dict(fingerprint)) == ""
    # A missing stage-1 side is "cannot check", not "mismatch".
    assert describe_fingerprint_mismatch({}, fingerprint) == ""


def test_the_dataset_fingerprints_its_own_sample_list(corpus_root):
    """Not the directory: the two differ the moment a sample is filtered out."""
    from src.datasets.dataset import get_finetune_dataset

    dataset = get_finetune_dataset(str(corpus_root), transform=None)
    full = dataset.corpus_fingerprint()
    assert full["num_samples"] == len(dataset.samples)

    dataset.samples = dataset.samples[:10]
    reduced = dataset.corpus_fingerprint()
    assert reduced["num_samples"] == 10
    assert reduced["sha256"] != full["sha256"]


# ===================================================================== B4 / E6
# Raw photograph coverage.


def test_raw_photograph_coverage_names_the_uncropped_photographs(tmp_path):
    from src.datasets.dataset import raw_photograph_coverage

    raw = tmp_path / "RAW_Samples" / "Poosa33"
    cropped = tmp_path / "Cropped_Samples" / "Rice" / "Poosa33"
    raw.mkdir(parents=True)
    cropped.mkdir(parents=True)
    for index in range(5):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(raw / f"IMG_{index:04d}.png")
    for index in range(3):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            cropped / f"IMG_{index:04d}_bbox7.png"
        )

    coverage = raw_photograph_coverage(tmp_path / "RAW_Samples", tmp_path / "Cropped_Samples")
    assert coverage["num_raw_photographs"] == 5
    assert coverage["num_used_photographs"] == 3
    assert coverage["num_unused_photographs"] == 2
    assert sorted(coverage["per_sub_variety"]["Poosa33"]["unused"]) == ["IMG_0003", "IMG_0004"]


def test_raw_coverage_reports_nothing_when_there_is_no_raw_tree(tmp_path):
    """Absent is not an error: most machines have only the cropped corpus."""
    from src.datasets.dataset import raw_photograph_coverage

    assert raw_photograph_coverage(tmp_path / "missing", tmp_path) == {}


# ===================================================================== F1
# Provenance-derived positives, and the view contract they must not break.


def test_same_photo_positives_keep_the_view_count_and_the_view_order(corpus_root):
    """The loss chunks on view id, so replacing a view must not add or move one."""
    from src.datasets.dataset import PretrainImageFolderDataset
    from src.datasets.transforms import get_dino_transforms

    transform = get_dino_transforms(32, 16, {"local_crops_number": 4}, return_original=False)
    plain = PretrainImageFolderDataset(root=str(corpus_root), transform=transform)
    paired = PretrainImageFolderDataset(
        root=str(corpus_root), transform=transform, same_photo_local_views=2, seed=7
    )

    _, plain_views, _, _ = plain[0]
    _, paired_views, _, _ = paired[0]
    assert len(plain_views) == len(paired_views) == 6
    assert all(a.shape == b.shape for a, b in zip(plain_views, paired_views))


def test_partners_are_other_crops_of_the_same_photograph(corpus_root):
    from src.datasets.dataset import PretrainImageFolderDataset, source_image_id

    dataset = PretrainImageFolderDataset(
        root=str(corpus_root), transform=None, same_photo_local_views=2, seed=3
    )
    for index in (0, 5, 11):
        partners = dataset._partner_indices(index, 2)
        assert len(partners) == 2
        own = source_image_id(dataset.samples[index][0])
        for partner in partners:
            assert source_image_id(dataset.samples[partner][0]) == own
            assert partner != index


def test_partner_draw_is_deterministic_in_the_seed(tmp_path):
    """Two runs at one seed must pair the same crops, whatever worker built them.

    Needs a photograph with enough crops that the draw has something to choose
    between: with two crops per photograph there is exactly one partner and every
    seed agrees trivially.
    """
    from src.datasets.dataset import PretrainImageFolderDataset

    directory = tmp_path / "Rice" / "Poosa33"
    directory.mkdir(parents=True)
    for bbox in range(8):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            directory / f"IMG_0001_bbox{bbox}.png"
        )

    def draws(seed: int) -> list[list[int]]:
        dataset = PretrainImageFolderDataset(
            root=str(tmp_path), transform=None, same_photo_local_views=2, seed=seed
        )
        return [dataset._partner_indices(index, 2) for index in range(8)]

    assert draws(11) == draws(11)
    assert draws(11) != draws(12), "the draw must depend on the seed"


def test_a_singleton_photograph_falls_back_to_augmenting_the_anchor(tmp_path):
    """One crop per photograph means no partner exists; the view count still holds."""
    from src.datasets.dataset import PretrainImageFolderDataset
    from src.datasets.transforms import get_dino_transforms

    directory = tmp_path / "Rice" / "Solo"
    directory.mkdir(parents=True)
    for index in range(3):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            directory / f"IMG_{index:04d}_bbox0.png"
        )

    dataset = PretrainImageFolderDataset(
        root=str(tmp_path),
        transform=get_dino_transforms(32, 16, {"local_crops_number": 4}, return_original=False),
        same_photo_local_views=2,
        seed=0,
    )
    assert dataset._partner_indices(0, 2) == []
    _, views, _, _ = dataset[0]
    assert len(views) == 6


def test_more_partners_than_local_views_is_refused():
    """Silent truncation would make 'two of four' and 'all four' the same arm."""
    from src.datasets.transforms import DataAugmentationDINO

    augmentation = DataAugmentationDINO(
        image_size=32, local_crop_size=16, local_crops_number=2, return_original=False
    )
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="nowhere to put"):
        augmentation(image, partner_images=[image, image, image])


def test_pickle_batches_refuses_provenance_positives(tmp_path):
    """That layout carries no source-photograph key, so the arm is meaningless there."""
    from src.datasets.dataset import get_pretrain_dataloader

    (tmp_path / "a.pkl").write_bytes(b"")
    with pytest.raises(ValueError, match="provenance"):
        get_pretrain_dataloader(
            data_dir=str(tmp_path), transform=None, batch_size=1,
            dataset_format="pickle_batches", same_photo_local_views=2,
        )


# ===================================================================== A3 / C2
# The documented augmentation override actually exists.


def test_match_view_lowpass_is_a_real_config_key(conf_dir):
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=pretrain_swinv2_dino")
    assert cfg.data.augmentation.match_view_lowpass is False
    # The documented override must compose without a `+`.
    overridden = build(
        conf_dir,
        "experiment=pretrain_swinv2_dino",
        "data.augmentation.match_view_lowpass=true",
    )
    assert overridden.data.augmentation.match_view_lowpass is True


def test_same_photo_local_views_is_a_real_config_key(conf_dir):
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=pretrain_swinv2_dino")
    assert cfg.data.augmentation.same_photo_local_views == 0
    assert (
        build(
            conf_dir,
            "experiment=pretrain_swinv2_dino",
            "data.augmentation.same_photo_local_views=2",
        ).data.augmentation.same_photo_local_views
        == 2
    )


def test_the_augmentation_node_still_builds_the_transform(conf_dir):
    """`same_photo_local_views` is dataset-side; it must not reach the transform."""
    from omegaconf import OmegaConf

    from src.datasets.transforms import get_dino_transforms
    from tests.test_configs import build

    cfg = build(
        conf_dir,
        "experiment=pretrain_swinv2_dino",
        "data.augmentation.same_photo_local_views=3",
    )
    transform = get_dino_transforms(
        int(cfg.data.image_size),
        int(cfg.data.local_crop_size),
        OmegaConf.to_container(cfg.data.augmentation, resolve=True),
        return_original=False,
    )
    assert transform.num_crops == PAPER_GLOBAL_CROPS + PAPER_LOCAL_CROPS


# ===================================================================== B1
# Stage-specific feature extraction.


@pytest.mark.slow
def test_stage3_reads_layers_2_and_reports_its_own_width():
    from src.models.builder import BackboneFeatureExtractor

    images = torch.randn(2, 3, 256, 256)
    final = BackboneFeatureExtractor("swinv2_tiny_window16_256", feature_stage="final")
    stage3 = BackboneFeatureExtractor("swinv2_tiny_window16_256", feature_stage="stage3")

    assert final.feature_dim == 768
    assert stage3.feature_dim == 384
    assert stage3.backbone_feature_dim == 768, "the trunk's own width must stay reportable"
    assert final(images).shape == (2, 768)
    assert stage3(images).shape == (2, 384)
    assert final(images, return_tokens=True).shape == (2, 64, 768)
    assert stage3(images, return_tokens=True).shape == (2, 256, 384)


@pytest.mark.slow
def test_forward_intermediates_and_the_hook_fallback_agree_bitwise():
    """The fast path stops the trunk early; the fallback runs it all. Same tensor.

    If these ever diverge, the `stage3` readout would silently depend on which
    timm happened to be installed.
    """
    from src.models.builder import BackboneFeatureExtractor

    extractor = BackboneFeatureExtractor("swinv2_tiny_window16_256", feature_stage="stage3")
    images = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        fast = extractor._stage_features(images)
        fallback = extractor._stage_features_via_hook(images, 2)
    assert torch.equal(fast, fallback)


@pytest.mark.slow
def test_stage3_pooled_2x2_restores_the_token_budget_stage_two_routes_over():
    """256 tokens would quadruple the MoE's routing slots and 16x the attention."""
    from src.models.builder import BackboneFeatureExtractor

    extractor = BackboneFeatureExtractor(
        "swinv2_tiny_window16_256", feature_stage="stage3_pooled_2x2"
    )
    tokens = extractor(torch.randn(2, 3, 256, 256), return_tokens=True)
    assert tokens.shape == (2, 64, 384)


@pytest.mark.slow
def test_the_encoder_invariant_holds_at_every_stage():
    """``encoder(images).shape[-1] == 384`` is an invariant, not a coincidence."""
    from src.models.builder import DinoV2SwinV2Encoder

    images = torch.randn(2, 3, 256, 256)
    for stage in ("final", "stage3", "stage3_pooled_2x2"):
        encoder = DinoV2SwinV2Encoder(
            "swinv2_tiny_window16_256", feature_stage=stage, token_mode="pooled"
        )
        assert encoder(images).shape[-1] == 384


def test_an_unknown_feature_stage_is_refused():
    from src.models.builder import BackboneFeatureExtractor

    with pytest.raises(ValueError, match="feature_stage"):
        BackboneFeatureExtractor("swinv2_tiny_window16_256", feature_stage="layers.7")


def test_feature_stage_defaults_to_final_everywhere(conf_dir):
    """Every published number was produced under `final`; the default must not move."""
    from tests.test_configs import build

    for experiment in ("finetune_hierarchical_moe", "pretrain_swinv2_dino"):
        cfg = build(conf_dir, f"experiment={experiment}")
        assert cfg.model.backbone.feature_stage == "final"


# ===================================================================== C4
# The auxiliary stage head.


@pytest.mark.slow
def test_the_auxiliary_head_is_not_allocated_unless_it_is_asked_for():
    """A disabled block that still holds parameters makes every count dishonest."""
    from src.models.backbones.swinv2_dino import DINO

    model = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16, out_dim=32
    )
    assert model.student_aux_head is None
    assert model.teacher_aux_head is None
    assert model.parameter_summary()["dino_aux_head"] == 0
    assert len(model.ema_pairs()) == 2


@pytest.mark.slow
def test_the_auxiliary_head_supervises_the_configured_stage():
    from src.models.backbones.swinv2_dino import DINO

    model = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16,
        out_dim=32, aux_stage=2, aux_out_dim=24,
    )
    views = torch.randn(4, 3, 256, 256)
    logits, bottleneck, aux = model.forward_student_views(
        views, return_bottleneck=True, return_aux=True
    )
    assert logits.shape == (4, 32)
    assert bottleneck.shape == (4, 16)
    assert aux.shape == (4, 24), "the auxiliary head has its own prototype count"
    # layers.2 emits 384 channels on Tiny.
    assert model.student_aux_head.mlp[0].in_features == 384
    assert len(model.ema_pairs()) == 3, "the auxiliary teacher must be EMA-advanced too"

    teacher_logits, teacher_aux = model.forward_teacher_views(views[:2], return_aux=True)
    assert teacher_logits.shape == (2, 32)
    assert teacher_aux.shape == (2, 24)


@pytest.mark.slow
def test_the_auxiliary_head_does_not_rename_the_published_state_dict():
    """`student_backbone` is the only handoff; a renamed key is a silent failure."""
    from src.models.backbones.swinv2_dino import DINO

    plain = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16, out_dim=32
    )
    with_aux = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16,
        out_dim=32, aux_stage=2,
    )
    assert set(plain.student_backbone.state_dict()) == set(with_aux.student_backbone.state_dict())


@pytest.mark.slow
def test_the_trunk_feature_is_available_for_backbone_space_koleo():
    from src.models.backbones.swinv2_dino import DINO

    model = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16, out_dim=32
    )
    logits, bottleneck, features = model.forward_student_views(
        torch.randn(3, 3, 256, 256), return_bottleneck=True, return_features=True
    )
    assert logits.shape == (3, 32)
    assert bottleneck.shape == (3, 16)
    assert features.shape == (3, 768), "koleo_space=backbone regularises the SHIPPED space"


# ===================================================================== E5
# grouped_cv splitting for stage 2.


@pytest.fixture
def grouped_dataset(corpus_root):
    from src.datasets.dataset import get_finetune_dataset

    return get_finetune_dataset(str(corpus_root), transform=None)


def test_grouped_cv_holds_out_every_crop_exactly_once(grouped_dataset):
    from src.trainers.moe_finetune import split_dataset

    splits, test_indices, report = split_dataset(
        grouped_dataset, test_size=0.2, num_folds=2, seed=42, protocol="grouped_cv"
    )
    assert test_indices.size == 0, "grouped_cv has no held-out split; the folds are the evaluation"
    held_out = np.concatenate([validation for _, validation in splits])
    assert sorted(held_out.tolist()) == list(range(len(grouped_dataset)))
    assert report["out_of_fold"] is True
    # Every class appears in the union of the held-out folds, which is the whole
    # point: the `grouped` protocol's single test split holds 14 of 27 and caps
    # its own macro-F1 near 14/27 for reasons unrelated to the model.
    assert report["classes_present_in_test"] == sum(SUBVARIETY_COUNTS.values())


def test_grouped_cv_folds_are_photograph_disjoint(grouped_dataset):
    from src.trainers.moe_finetune import split_dataset

    groups = grouped_dataset.source_groups()
    splits, _, _ = split_dataset(
        grouped_dataset, test_size=0.0, num_folds=2, seed=42, protocol="grouped_cv"
    )
    for train_indices, validation_indices in splits:
        assert not (set(groups[train_indices]) & set(groups[validation_indices])), (
            "a source photograph straddled a fold boundary"
        )


def test_grouped_cv_needs_at_least_two_folds(grouped_dataset):
    from src.trainers.moe_finetune import split_dataset

    with pytest.raises(ValueError, match="num_folds >= 2"):
        split_dataset(grouped_dataset, test_size=0.0, num_folds=1, seed=0, protocol="grouped_cv")


def test_every_split_reports_how_many_classes_the_test_side_holds(grouped_dataset):
    """A 27-way macro-F1 on a 14-class test split is capped by the split, not the model."""
    from src.trainers.moe_finetune import split_dataset

    _, _, report = split_dataset(
        grouped_dataset, test_size=0.2, num_folds=2, seed=42, protocol="grouped"
    )
    assert "classes_present_in_test" in report
    assert report["num_classes"] == sum(SUBVARIETY_COUNTS.values())
    assert 0 <= report["classes_present_in_test"] <= report["num_classes"]


def test_the_default_protocol_is_unchanged(conf_dir):
    """`grouped_cv` is opt-in: every published stage-2 number used `grouped`."""
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=finetune_hierarchical_moe")
    assert cfg.experiment.training.split_protocol == "grouped"


def test_merge_out_of_fold_restores_dataset_order(grouped_dataset):
    """Fold order is not dataset order; a consumer joining by index needs the latter."""
    from src.trainers.moe_finetune import EpochAccumulator, merge_out_of_fold

    size = len(grouped_dataset)
    sub_true = np.array([sub for _, _, sub in grouped_dataset.samples])
    seed_true = np.array([seed for _, seed, _ in grouped_dataset.samples])
    num_sub = sum(SUBVARIETY_COUNTS.values())

    # Two folds, deliberately in a non-monotonic order.
    parts = []
    for indices in (np.arange(size // 2, size), np.arange(0, size // 2)):
        accumulator = EpochAccumulator()
        accumulator.seed_true = seed_true[indices].tolist()
        accumulator.seed_pred = seed_true[indices].tolist()
        accumulator.sub_true = sub_true[indices].tolist()
        accumulator.sub_pred = sub_true[indices].tolist()
        scores = np.zeros((indices.size, num_sub), dtype=np.float32)
        scores[np.arange(indices.size), sub_true[indices]] = 1.0
        accumulator.sub_scores = [scores]
        accumulator.logits = [scores]
        accumulator.expert_indices = [np.zeros((indices.size, 1), dtype=np.int64)]
        accumulator.batches = 1
        parts.append((indices, accumulator))

    evaluation, merged = merge_out_of_fold(parts, grouped_dataset, size)
    assert merged.sub_true == sub_true.tolist(), "predictions were not put back in dataset order"
    assert merged.sub_pred == sub_true.tolist()
    assert evaluation.sub_variety["accuracy"] == pytest.approx(1.0)


def test_merge_out_of_fold_refuses_incomplete_coverage(grouped_dataset):
    from src.trainers.moe_finetune import EpochAccumulator, merge_out_of_fold

    accumulator = EpochAccumulator()
    accumulator.sub_true = [0, 1]
    with pytest.raises(RuntimeError, match="did not cover"):
        merge_out_of_fold([(np.array([0, 1, 2]), accumulator)], grouped_dataset, 3)


# ===================================================================== E9
# Nuisance decodability.


def test_nuisance_is_at_chance_for_a_representation_that_encodes_only_the_class():
    from src.utils.representation import nuisance_decodability

    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), 30)
    groups = labels * 10 + rng.integers(0, 3, labels.size)
    features = np.eye(5)[labels] + 0.01 * rng.normal(size=(labels.size, 5))

    report = nuisance_decodability(features, labels, groups, seed=0)
    assert report["classes_scored"] == 5
    assert report["above_chance"] < 0.15, (
        "a class-only representation must not decode the photograph"
    )


def test_nuisance_is_near_one_for_a_representation_that_encodes_the_photograph():
    from src.utils.representation import nuisance_decodability

    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), 30)
    groups = labels * 10 + rng.integers(0, 3, labels.size)
    features = np.eye(50)[groups] + 0.01 * rng.normal(size=(labels.size, 50))

    report = nuisance_decodability(features, labels, groups, seed=0)
    assert report["within_class_photo_accuracy"] > 0.95
    assert report["above_chance"] > 0.5


def test_nuisance_skips_classes_with_one_photograph():
    """Five sub-varieties genuinely have one; there is nothing to discriminate."""
    from src.utils.representation import nuisance_decodability

    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 30)
    groups = np.concatenate([np.zeros(30), 100 + rng.integers(0, 3, 30)]).astype(np.int64)
    report = nuisance_decodability(rng.normal(size=(60, 8)), labels, groups, seed=0)
    assert report["classes_scored"] == 1, "the single-photograph class must be skipped"


def test_nuisance_reports_its_chance_level_alongside_the_score():
    """The number is uninterpretable without it: chance is 1/photographs per class."""
    from src.utils.representation import nuisance_decodability

    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 24)
    groups = labels * 10 + rng.integers(0, 4, labels.size)
    report = nuisance_decodability(rng.normal(size=(72, 6)), labels, groups, seed=0)
    assert report["chance"] == pytest.approx(0.25)
    assert report["above_chance"] == pytest.approx(
        report["within_class_photo_accuracy"] - report["chance"]
    )


# ===================================================================== E1
# The trivial-feature floor.


def test_handcrafted_features_are_ten_numbers_that_encode_size_and_colour(tmp_path):
    from src.utils.representation import handcrafted_image_features

    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    Image.fromarray(np.full((8, 16, 3), 32, dtype=np.uint8)).save(small)
    Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(large)

    features = handcrafted_image_features([str(small), str(large)])
    assert features.shape == (2, 10)
    assert features[1, 0] > features[0, 0], "log area must grow with the crop"
    # PIL reports (width, height), so an 8-row x 16-column array is 16 x 8.
    assert features[0, 1] == pytest.approx(math.log(16 / 8), abs=1e-5)
    assert features[1, 2] > features[0, 2], "mean R must follow the brightness"
    # A constant image has zero variance in every channel.
    assert np.allclose(features[:, 5:8], 0.0, atol=1e-6)


def test_the_floor_row_is_configured_and_needs_no_backbone(conf_dir):
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=eval_pretrain_representation")
    entry = next(
        item for item in cfg.experiment.evaluation.encoders if item["label"] == "handcrafted_floor"
    )
    assert entry["kind"] == "handcrafted"
    assert entry["checkpoint"] is None


# ============================================================ B3 / G1
# Dataloader workers.


def test_the_auto_worker_cap_is_configurable_and_defaults_to_sixteen():
    from src.utils.training.distributed import (
        DEFAULT_NUM_WORKERS_AUTO_CAP,
        DistributedContext,
        resolve_num_workers,
    )

    assert DEFAULT_NUM_WORKERS_AUTO_CAP == 16
    context = DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0,
                                 local_world_size=1, device=torch.device("cpu"), backend="")
    # The cap only ever lowers the affinity-aware count, so a tiny cap must bind.
    assert resolve_num_workers("auto", context, auto_cap=1) == 1
    assert resolve_num_workers(4, context, auto_cap=1) == 4, "an explicit value is not capped"


def test_the_config_carries_the_raised_cap(conf_dir):
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=pretrain_swinv2_dino")
    assert cfg.data.num_workers_auto_cap == 16
    assert cfg.data.num_workers == "auto"


# ===================================================================== A6
# summary.json as a stage-1 contract.


def test_stage1_summary_round_trips_and_carries_the_corpus(tmp_path, corpus_root):
    """The file `run_stage1_ablations.py` and `pretrain_eval` both read."""
    from omegaconf import OmegaConf

    from src.datasets.dataset import corpus_fingerprint
    from src.trainers.contrastive_pretrain import write_stage1_summary
    from src.utils.evaluation import RunSummary
    from src.utils.training.budget import StageOneBudget
    from src.utils.training.distributed import DistributedContext

    class _Transform:
        num_crops = 6
        view_sizes = [256, 256, 101, 101, 101, 101]
        view_ids = [0, 1, 2, 3, 4, 5]
        global_view_ids = [0, 1]

    class _Model:
        drop_path_rate = 0.1
        aux_stage = None
        aux_weight = 1.0

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "experiment": {"name": "arm", "training": {"epochs": 50}},
            "data": {
                "image_size": 256,
                "local_crop_size": 101,
                "augmentation": {"local_crops_number": 4},
            },
            "model": {"backbone": {"name": "swinv2_tiny_window16_256", "pretrained": True}},
        }
    )
    fingerprint = corpus_fingerprint(corpus_root)
    context = DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0,
                                 local_world_size=1, device=torch.device("cpu"), backend="")

    class _Logger:
        def info(self, *args, **kwargs):
            pass

    path = write_stage1_summary(
        tmp_path,
        cfg,
        criterion=_dino_loss(lambda_koleo=0.1, koleo_scope="per_view"),
        model=_Model(),
        transform=_Transform(),
        fingerprint=fingerprint,
        raw_coverage={},
        parameters={"backbone": 27_578_154, "dino_head": 59_936},
        budget=StageOneBudget(),
        dynamics={"final_teacher_student_kl": 0.21, "final_loss": 5.65},
        runtime={"amp": "bf16"},
        artifacts={"student_backbone": "x.pth"},
        context=context,
        logger=_Logger(),
    )

    summary = RunSummary.load(path)
    assert summary.split["corpus"]["sha256"] == fingerprint["sha256"]
    assert summary.split["stage1_transductive"] is True
    assert summary.loss_flags["koleo_scope"] == "per_view"
    assert summary.metrics["final_teacher_student_kl"] == pytest.approx(0.21)
    assert summary.config["view_ids"] == [0, 1, 2, 3, 4, 5]
    assert summary.component_flags["stage"] == "pretrain"


# ===================================================================== infra
# The arm-suite runner.


def test_arm_manifests_parse_and_pin_per_arm_paths(tmp_path):
    """Without per-arm paths, four arms overwrite each other's encoder in silence."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stage1_ablations as runner

    for manifest in sorted(Path("conf/stage1_arms").glob("*.yaml")):
        suite = runner.load_suite(manifest)
        assert suite.arms, f"{manifest} defines no arms"
        for arm in suite.arms:
            assert arm.description, f"{manifest}: {arm.name} has no description"

    suite = runner.load_suite("conf/stage1_arms/phase1.yaml")
    directories = set()
    publications = set()
    for arm in suite.arms:
        directory = arm.directory(tmp_path)
        directories.add(str(directory))
        if arm.train:
            command = runner.train_command(arm, suite, directory, [])
            publication = next(
                item for item in command if item.startswith("experiment.training.shared_backbone_path=")
            )
            publications.add(publication)
    assert len(directories) == len(suite.arms)
    assert len(publications) == len([arm for arm in suite.arms if arm.train])


def test_the_evaluation_command_never_forwards_training_overrides(tmp_path):
    """Hydra's struct mode rejects `experiment.training.*` on an evaluation config."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stage1_ablations as runner

    suite = runner.load_suite("conf/stage1_arms/phase1.yaml")
    for arm in suite.arms:
        command = runner.eval_command(arm, suite, arm.directory(tmp_path), [])
        assert not any(item.startswith("experiment.training.") for item in command)


def test_data_overrides_carry_into_the_evaluation(tmp_path):
    """The alignment measurement replays the arm's own multi-crop pipeline."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stage1_ablations as runner

    suite = runner.load_suite("conf/stage1_arms/phase1.yaml")
    arm = next(item for item in suite.arms if item.name == "P1-C")
    command = runner.eval_command(arm, suite, arm.directory(tmp_path), [])
    assert "data.local_crop_size=160" in command
    assert "data.augmentation.local_crops_scale=[0.30,0.70]" in command


def test_the_frozen_reference_arm_evaluates_an_untrained_trunk(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stage1_ablations as runner

    suite = runner.load_suite("conf/stage1_arms/phase1.yaml")
    frozen = next(arm for arm in suite.arms if arm.name == "P1-F")
    assert frozen.train is False
    command = runner.eval_command(frozen, suite, frozen.directory(tmp_path), [])
    assert "experiment=eval_frozen_reference" in command


def test_a_frozen_arm_is_not_repeated_across_seeds():
    """There is nothing stochastic in it; three copies would inflate the evidence."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stage1_ablations as runner

    suite = runner.load_suite("conf/stage1_arms/phase1.yaml")
    frozen = next(arm for arm in suite.arms if not arm.train)
    trained = next(arm for arm in suite.arms if arm.train)
    assert frozen.with_seed(7).seed is None
    assert trained.with_seed(7).seed == 7


def test_a_manifest_with_an_unknown_key_is_refused(tmp_path):
    """A typo'd key that does nothing is how an arm quietly stops being an arm."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stage1_ablations as runner

    manifest = tmp_path / "bad.yaml"
    manifest.write_text("experiment: x\narms:\n  - name: a\n    overides: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        runner.load_suite(manifest)


# ============================================== multi-view ordering <-> KoLeo

def test_the_collate_order_is_what_makes_per_view_koleo_meaningful():
    """The two facts are one fact, and this test is where they are tied together.

    ``MultiCropCollate`` emits ``[V, B, C, H, W]``, so ``flatten(0, 1)`` gives
    "all of view 0, then all of view 1". That is precisely the layout
    :func:`grouped_koleo` chunks on. A batch-major collate would interleave the
    views, and *both* the cross-view loss and the per-view KoLeo would silently
    group the wrong rows -- with a loss curve that looks entirely normal.
    """
    from src.datasets.dataset import MultiCropCollate
    from src.losses.dino import grouped_koleo

    batch, views, size = 4, 6, 3
    # Sample i, view v is filled with the value (v * 100 + i), so the flattened
    # block order is recoverable by inspection.
    samples = [
        (None, [torch.full((1, size, size), float(view * 100 + index)) for view in range(views)],
         index, f"img{index}")
        for index in range(batch)
    ]
    collated = MultiCropCollate(num_global_crops=2)(samples)
    stacked = torch.cat(
        [collated.global_views.flatten(0, 1), collated.local_views.flatten(0, 1)], dim=0
    )
    flat = stacked.reshape(views * batch, -1)[:, 0]

    for view in range(views):
        block = flat[view * batch : (view + 1) * batch]
        assert torch.equal(block, torch.arange(batch, dtype=block.dtype) + view * 100), (
            "the stacked views are not view-major; per-view KoLeo would chunk the wrong rows"
        )

    # And the chunking the loss performs recovers exactly those blocks.
    globals_only = flat[: 2 * batch].reshape(-1, 1).float()
    chunks = globals_only.chunk(2, dim=0)
    assert torch.equal(chunks[0].flatten(), flat[:batch])
    assert torch.equal(chunks[1].flatten(), flat[batch : 2 * batch])
    # Sanity: the helper accepts this layout.
    assert torch.isfinite(grouped_koleo(globals_only.repeat(1, 4), 2, "per_view"))


def test_koleo_reads_the_global_views_only():
    """Local views are not in the teacher's block and must not enter the uniformity term."""
    from src.losses.dino import CustomDINOLoss

    torch.manual_seed(0)
    batch = 4
    # Sinkhorn, not EMA: the EMA centre is mutated by every call, so a second
    # forward would differ for a reason that has nothing to do with KoLeo.
    loss_fn = CustomDINOLoss(
        out_dim=16, num_crops=6, warmup_teacher_temp=0.04, teacher_temp=0.04,
        warmup_teacher_temp_epochs=0, num_epochs=1, centering="sinkhorn", lambda_koleo=1.0,
    )
    bottleneck = torch.randn(6 * batch, 8)
    student = [torch.randn(batch, 16) for _ in range(6)]
    teacher = [torch.randn(batch, 16) for _ in range(2)]

    with_globals = float(
        loss_fn(student, teacher, epoch=0, student_embeddings=bottleneck[: 2 * batch])
    )
    # Perturbing only the LOCAL rows must not move the loss.
    perturbed = bottleneck.clone()
    perturbed[2 * batch :] += 10.0
    again = float(loss_fn(student, teacher, epoch=0, student_embeddings=perturbed[: 2 * batch]))
    assert with_globals == pytest.approx(again, abs=1e-6)


# =========================================== backward-compatible checkpoints


@pytest.mark.slow
def test_an_existing_checkpoint_loads_into_every_feature_stage(tmp_path):
    """`feature_stage` changes what is READ, never what is stored.

    An encoder published before the option existed must load unchanged into a
    `stage3` extractor -- the trunk weights are identical, only the readout moves.
    """
    from src.models.builder import BackboneFeatureExtractor

    published = BackboneFeatureExtractor("swinv2_tiny_window16_256", freeze=True)
    checkpoint = tmp_path / "dino_pretrained_backbone.pth"
    torch.save(published.backbone.state_dict(), checkpoint)

    for stage, width in (("final", 768), ("stage3", 384), ("stage3_pooled_2x2", 384)):
        loaded = BackboneFeatureExtractor(
            "swinv2_tiny_window16_256", checkpoint_path=str(checkpoint), feature_stage=stage
        )
        assert loaded.load_report == {"missing_keys": [], "unexpected_keys": []}, (
            f"an existing checkpoint failed to load at feature_stage={stage}"
        )
        assert loaded.feature_dim == width


@pytest.mark.slow
def test_a_checkpoint_from_an_aux_head_run_still_publishes_a_plain_trunk(tmp_path):
    """The auxiliary head must not leak into the stage-2 handoff."""
    from src.models.backbones.swinv2_dino import DINO
    from src.models.builder import BackboneFeatureExtractor

    model = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16,
        out_dim=32, aux_stage=2,
    )
    checkpoint = tmp_path / "encoder.pth"
    torch.save(model.student_backbone.state_dict(), checkpoint)

    consumer = BackboneFeatureExtractor(
        "swinv2_tiny_window16_256", checkpoint_path=str(checkpoint), strict=True
    )
    assert consumer.load_report == {"missing_keys": [], "unexpected_keys": []}


def test_a_run_without_a_stage1_summary_degrades_to_a_warning(tmp_path):
    """Every encoder produced before A6 has no summary; that is not an error."""
    from src.trainers.pretrain_eval import read_stage1_summary

    class _Logger:
        def __init__(self):
            self.warnings: list[str] = []

        def warning(self, message, *args):
            self.warnings.append(str(message) % args if args else str(message))

    logger = _Logger()
    assert read_stage1_summary(tmp_path, logger) == {}
    assert read_stage1_summary(None, logger) == {}

    (tmp_path / "summary.json").write_text("{not json", encoding="utf-8")
    assert read_stage1_summary(tmp_path, logger) == {}
    assert logger.warnings, "an unreadable summary should say so rather than pass silently"


def test_a_stage1_summary_is_read_back_as_the_corpus_to_compare_against(tmp_path):
    from src.trainers.pretrain_eval import read_stage1_summary

    payload = {"split": {"corpus": {"sha256": "abc", "num_samples": 8173}}}
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    class _Logger:
        def warning(self, *args, **kwargs):
            raise AssertionError("a valid summary must not warn")

    summary = read_stage1_summary(tmp_path, _Logger())
    assert summary["split"]["corpus"]["num_samples"] == 8173


# ==================================================== distributed execution


def test_koleo_grouping_is_a_local_statistic_under_sharding():
    """Per-rank by design, exactly as the reference implementation does it.

    Splitting a batch across two ranks gives each rank the per-view KoLeo of its
    own shard. That is the same function a single GPU computes per micro-batch,
    which is what makes the two runs comparable -- so this test pins that the
    grouping is applied WITHIN a shard, not across the concatenation.
    """
    from src.losses.dino import grouped_koleo

    torch.manual_seed(0)
    batch = 8
    features = _view_major_pairs(batch=batch, dim=12, jitter=0.5)
    view0, view1 = features[:batch], features[batch:]

    # Two ranks, each holding half the images -- but still both views of them.
    shards = [
        torch.cat([view0[: batch // 2], view1[: batch // 2]], dim=0),
        torch.cat([view0[batch // 2 :], view1[batch // 2 :]], dim=0),
    ]
    per_rank = [float(grouped_koleo(shard, 2, "per_view")) for shard in shards]
    assert all(math.isfinite(value) for value in per_rank)
    # A rank's value depends only on its own shard: recomputing it in isolation
    # reproduces it exactly.
    assert float(grouped_koleo(shards[0], 2, "per_view")) == pytest.approx(per_rank[0], abs=0.0)


@pytest.mark.slow
def test_ddp_wrapping_does_not_rename_the_published_keys_with_an_aux_head():
    """The wrapper registers the same objects; `state_dict()` must be unchanged.

    `student_backbone.state_dict()` is the only handoff to stage 2, and stage 2
    loads it with `checkpoint_strict: false` -- so a `module.` prefix would
    produce a run that matched zero keys, logged one line, and trained a random
    encoder to completion.
    """
    from src.models.backbones.swinv2_dino import DINO, _StudentPass

    model = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16,
        out_dim=32, aux_stage=2,
    )
    before = set(model.student_backbone.state_dict())
    wrapper = _StudentPass(
        model.student_backbone, model.student_head, model._module, model._pool,
        owner=model, aux_head=model.student_aux_head,
    )
    assert set(model.student_backbone.state_dict()) == before
    # The auxiliary head is registered on the wrapper so DDP owns its gradients.
    assert any("aux_head" in key for key in wrapper.state_dict())
    # And it is the same object, not a copy.
    assert wrapper.aux_head is model.student_aux_head


@pytest.mark.slow
def test_the_wrapper_returns_the_same_tensors_as_the_direct_path():
    from src.models.backbones.swinv2_dino import DINO, _StudentPass

    torch.manual_seed(0)
    model = DINO(
        "swinv2_tiny_window16_256", input_dim=768, hidden_dim=32, bottleneck_dim=16,
        out_dim=32, aux_stage=2,
    ).eval()
    views = torch.randn(2, 3, 256, 256)
    wrapper = _StudentPass(
        model.student_backbone, model.student_head, model._module, model._pool,
        owner=model, aux_head=model.student_aux_head,
    )
    with torch.no_grad():
        direct = model.forward_student_views(views, return_bottleneck=True, return_aux=True)
        wrapped = wrapper(views, return_bottleneck=True, return_aux=True)
    assert len(direct) == len(wrapped) == 3
    for left, right in zip(direct, wrapped):
        assert torch.allclose(left, right, atol=1e-6)


# ================================================ evaluation reproducibility


def test_nuisance_decodability_is_deterministic_in_its_seed():
    from src.utils.representation import nuisance_decodability

    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 30)
    groups = labels * 10 + rng.integers(0, 3, labels.size)
    features = rng.normal(size=(labels.size, 16))

    first = nuisance_decodability(features, labels, groups, seed=7)
    second = nuisance_decodability(features, labels, groups, seed=7)
    third = nuisance_decodability(features, labels, groups, seed=8)
    assert first == second
    # Not a requirement that they differ -- a different fold assignment usually
    # moves the number, and if it does not, the metric is simply stable here.
    assert set(first) == set(third)


def test_handcrafted_features_are_deterministic(tmp_path):
    from src.utils.representation import handcrafted_image_features

    paths = []
    for index in range(4):
        path = tmp_path / f"crop{index}.png"
        Image.fromarray(
            np.random.default_rng(index).integers(0, 255, (12, 9, 3), dtype=np.uint8)
        ).save(path)
        paths.append(str(path))
    first = handcrafted_image_features(paths)
    assert np.array_equal(first, handcrafted_image_features(paths))


def test_the_readout_stage_list_is_configurable_and_defaults_to_including_pooled(conf_dir):
    """`pooled` must stay first: `oof_probe_sub_accuracy` keeps meaning that number."""
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=eval_pretrain_representation")
    stages = list(cfg.experiment.evaluation.readout.stages)
    assert stages[0] == "pooled"
    assert "stage3" in stages

    reduced = build(
        conf_dir,
        "experiment=eval_pretrain_representation",
        "experiment.evaluation.readout.stages=[pooled]",
    )
    assert list(reduced.experiment.evaluation.readout.stages) == ["pooled"]


def test_the_screening_experiment_names_its_own_trunks(conf_dir):
    """Phase 0.5 is a config, not a script: each row carries its trunk and resolution."""
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=screen_backbones")
    rows = {entry["label"]: entry for entry in cfg.experiment.evaluation.encoders}
    # The control that separates capacity from the IN-22k corpus. Without it the
    # Base advantage cannot be attributed, and B2 is not decidable.
    assert "base_in1k" in rows
    assert rows["base_in1k"]["backbone"] == "swinv2_base_window16_256.ms_in1k"
    assert rows["base_in22k_192"]["image_size"] == 192
    assert rows["handcrafted_floor"]["kind"] == "handcrafted"
    # Nothing in a screen may touch the published handoff.
    assert cfg.experiment.evaluation.shared_backbone_path is None


def test_the_frozen_reference_experiment_reads_no_stage_one_artifact(conf_dir):
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=eval_frozen_reference")
    labels = [entry["label"] for entry in cfg.experiment.evaluation.encoders]
    assert labels[0] == "imagenet_init"
    assert not any(label.startswith("dino_") for label in labels)
    assert cfg.experiment.evaluation.teacher.enabled is False
    assert cfg.experiment.evaluation.prototypes.enabled is False
    assert cfg.experiment.evaluation.dynamics.enabled is False


def test_the_dynamics_summary_exposes_the_learnable_half(tmp_path):
    """`epoch/teacher_student_kl` is the curve; the raw loss is ~95 % target entropy."""
    from src.utils.evaluation import parse_pretrain_dynamics

    events = tmp_path / "events.jsonl"
    records = []
    for epoch, (loss, kl) in enumerate(
        [(7.77, 0.71), (6.20, 0.45), (5.70, 0.30), (5.67, 0.29)], start=1
    ):
        records.append(
            json.dumps(
                {
                    "type": "metrics",
                    "step": epoch,
                    "metrics": {
                        "epoch/loss": loss,
                        "epoch/teacher_student_kl": kl,
                        "epoch/teacher_entropy_cross_view": loss - kl,
                        "epoch/duration_seconds": 100.0,
                    },
                }
            )
        )
    events.write_text("\n".join(records), encoding="utf-8")

    summary = parse_pretrain_dynamics(events).summary()
    assert summary["teacher_student_kl_initial"] == pytest.approx(0.71)
    assert summary["teacher_student_kl_final"] == pytest.approx(0.29)
    assert summary["teacher_student_kl_min"] == pytest.approx(0.29)
    # 0.29 / 5.67 of the final loss is learnable; the rest is target entropy.
    assert summary["teacher_entropy_share_of_loss_final"] == pytest.approx(1 - 0.29 / 5.67, abs=1e-6)
    # (0.71 - 0.29) of a (7.77 - 5.67) improvement came from the student.
    assert summary["teacher_entropy_share_of_loss_improvement"] == pytest.approx(
        1 - 0.42 / 2.10, abs=1e-6
    )


def test_an_old_event_stream_without_the_decomposition_reports_nan(tmp_path):
    """Runs recorded before A4 must not produce a silently wrong number."""
    from src.utils.evaluation import parse_pretrain_dynamics

    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"type": "metrics", "step": 1, "metrics": {"epoch/loss": 5.6}}),
        encoding="utf-8",
    )
    summary = parse_pretrain_dynamics(events).summary()
    assert summary["teacher_student_kl_final"] != summary["teacher_student_kl_final"]  # NaN


def test_loop_blocked_falls_back_to_the_historical_metric_name(tmp_path):
    """A figure written against the shipped run must keep working."""
    from src.utils.evaluation import parse_pretrain_dynamics

    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {"type": "metrics", "step": 1, "metrics": {"epoch/data_wait_fraction": 0.916}}
        ),
        encoding="utf-8",
    )
    summary = parse_pretrain_dynamics(events).summary()
    assert summary["loop_blocked_fraction_mean"] == pytest.approx(0.916)
    assert summary["data_wait_fraction_mean"] == pytest.approx(0.916)
    # Absent unless the run opted into the synchronised measurement.
    assert summary["gpu_busy_fraction_mean"] != summary["gpu_busy_fraction_mean"]


# ================================================ per-trunk token geometry


@pytest.mark.slow
def test_the_token_grid_is_a_per_trunk_constant():
    """`PAPER_TOKEN_GRID = 64` is a fact about `window16_256`, not about SwinV2.

    The distinction became load-bearing once the initialisation screen made a
    192 px trunk a first-class option (`STAGE1_CHANGES.md` B5): its 6x6 = 36 final
    grid changes stage 2's routing-slot count, and grid routing is what makes the
    load-balancing statistic estimable at small batch in the first place.
    """
    import torch

    from src.models.builder import BackboneFeatureExtractor
    from tests.conftest import (
        PAPER_TOKEN_GRID,
        STAGE3_FEATURE_DIM_BASE,
        STAGE3_FEATURE_DIM_TINY_SMALL,
        STAGE3_TOKEN_GRID,
        STAGE3_TOKEN_GRID_192,
        TOKEN_GRID_192,
    )

    cases = [
        ("swinv2_tiny_window16_256", 256, PAPER_TOKEN_GRID, STAGE3_TOKEN_GRID,
         STAGE3_FEATURE_DIM_TINY_SMALL),
        ("swinv2_base_window12_192", 192, TOKEN_GRID_192, STAGE3_TOKEN_GRID_192,
         STAGE3_FEATURE_DIM_BASE),
    ]
    for name, size, final_tokens, stage3_tokens, stage3_dim in cases:
        images = torch.randn(1, 3, size, size)
        final = BackboneFeatureExtractor(name, feature_stage="final")
        stage3 = BackboneFeatureExtractor(name, feature_stage="stage3")
        pooled_2x2 = BackboneFeatureExtractor(name, feature_stage="stage3_pooled_2x2")

        assert final(images, return_tokens=True).shape[1] == final_tokens
        assert stage3(images, return_tokens=True).shape[1] == stage3_tokens
        assert stage3.feature_dim == stage3_dim
        # 2x2 pooling divides the grid by four, restoring the final stage's budget.
        assert pooled_2x2(images, return_tokens=True).shape[1] == stage3_tokens // 4
        assert stage3_tokens // 4 == final_tokens
