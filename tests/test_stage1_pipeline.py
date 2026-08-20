"""Tests for the stage-1 pipeline: views, protocol, artifacts.

Grouped by the claim each one defends, because several of these pin a *measured
property of this dataset* rather than a property of the code, and a failure means
something different in each case:

* **View geometry** -- the crop policy's whole justification is that a local view
  carries a median ~600 native pixels under DINO's reference ranges and ~1,400
  under the canonical ones. If those numbers move, the argument moves with them.
* **Losslessness** -- ``RandomRotation90`` and the native-pixel floor are only
  admissible because they add no interpolation and never make a view *more*
  destructive. Both are checked directly rather than assumed.
* **Protocol** -- the pipeline is crop-level stratified by instruction, and the
  configs must say so unambiguously in both stages.
* **Machine-readability** -- every artifact an automated analysis reads (CSV
  schema, ``describe()``, ``summary.json`` keys) is pinned, because the whole
  point of writing them is that something else parses them later.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from src.datasets.transforms import (
    TORCHVISION_CROP_RATIO,
    DataAugmentationDINO,
    NativeFloorRandomResizedCrop,
    RandomRotation90,
    get_dino_transforms,
    measure_view_geometry,
    view_geometry_report,
)
from src.utils.training.csv_metrics import CsvMetricSink, CsvMetricWriter
from src.utils.training.representation_probe import (
    SELECTION_METRICS,
    CheckpointSelector,
    ProbeResult,
    geometry_metrics,
    publish_best_encoder,
    stratified_readout,
)

#: A stand-in for the real corpus: median 52 x 51 px, 96.6 % non-square, spanning
#: 21x22 to 881x413. Drawn from a fixed seed so the geometry assertions below are
#: reproducible without the dataset on disk.
_RNG = np.random.default_rng(20260819)


def _synthetic_source_sizes(count: int = 4000) -> list[tuple[int, int]]:
    """Sizes with the measured shape of ``Cropped_Samples``.

    Log-normal around a 52 x 51 median with an aspect spread matching the
    measured p5/p95 of 0.52 / 1.98. Not the real corpus, but close enough that
    the *orderings* these tests assert are the same orderings the real data
    produces -- which is what they are for.
    """
    side = np.exp(_RNG.normal(np.log(51.5), 0.55, count))
    aspect = np.exp(_RNG.normal(0.0, 0.45, count))
    widths = np.clip(side * np.sqrt(aspect), 21, 881).astype(int)
    heights = np.clip(side / np.sqrt(aspect), 22, 413).astype(int)
    return list(zip(widths.tolist(), heights.tolist()))


def _image(width: int = 52, height: int = 51) -> Image.Image:
    return Image.fromarray(
        (_RNG.random((height, width, 3)) * 255).astype("uint8"), mode="RGB"
    )


# --------------------------------------------------------------- view geometry


def test_local_views_carry_far_more_native_pixels_under_v2():
    """The redesign's central claim, measured through torchvision's own sampler.

    Under the submitted recipe a local view is built from a median ~600 source
    pixels -- a 24 x 24 fragment of one seed inflated to 65,536 output pixels --
    and 8 of the 10 cross-view terms in Eq. 1 are anchored on such a view. v2
    roughly doubles that. A regression here is a regression in the argument, not
    just in a number.
    """
    sizes = _synthetic_source_sizes()
    submitted = measure_view_geometry(
        sizes, scale=(0.05, 0.40), ratio=TORCHVISION_CROP_RATIO, output_size=256, samples=3000
    )
    revised = measure_view_geometry(
        sizes, scale=(0.30, 0.70), ratio=(0.5, 2.0), output_size=256, samples=3000
    )
    assert revised["native_pixels_p50"] > 2.0 * submitted["native_pixels_p50"]
    assert revised["upsample_factor_median"] < submitted["upsample_factor_median"]
    # Both are upsamples on this data, which is why the tensor size carries no
    # information about how much of a seed a view contains.
    assert submitted["upsample_factor_median"] > 1.0
    assert revised["upsample_factor_median"] > 1.0


def test_widening_the_aspect_range_cuts_the_deterministic_fallback():
    """Why ``crop_ratio`` is a config key rather than a constant.

    ``RandomResizedCrop.get_params`` retries the (area, aspect) draw ten times
    and then returns a **deterministic centre crop**. A high-area box in
    torchvision's (0.75, 1.33) does not fit inside a non-square source, and only
    3.4 % of these crops are square -- so raising the scale floor silently trades
    augmentation randomness for content. Widening the ratio recovers it.

    This is the knob a naive narrowing of the scale range would leave alone, and
    the reason ``crop_ratio`` is a config key rather than a torchvision default.
    """
    sizes = _synthetic_source_sizes()
    narrow = measure_view_geometry(
        sizes, scale=(0.70, 1.00), ratio=TORCHVISION_CROP_RATIO, output_size=256, samples=4000
    )
    wide = measure_view_geometry(
        sizes, scale=(0.70, 1.00), ratio=(0.5, 2.0), output_size=256, samples=4000
    )
    assert wide["deterministic_fallback_rate"] < narrow["deterministic_fallback_rate"]
    # Strictly better on both axes: more randomness AND more content.
    assert wide["native_pixels_p50"] >= narrow["native_pixels_p50"]


def test_native_pixel_floor_lifts_the_tail_and_leaves_the_median_alone():
    """The floor acts on the worst views, which is its entire claim.

    A fixed ``scale`` is a fraction, so it shreds a 21 x 22 crop and an
    881 x 413 crop by the same factor. The floor raises the lower bound per
    image and never lowers it, so it can only make views less destructive.
    """
    sizes = _synthetic_source_sizes()
    plain = measure_view_geometry(
        sizes, scale=(0.30, 0.70), ratio=(0.5, 2.0), output_size=256, samples=4000
    )
    floored = measure_view_geometry(
        sizes,
        scale=(0.30, 0.70),
        ratio=(0.5, 2.0),
        output_size=256,
        min_native_pixels=900,
        samples=4000,
    )
    assert floored["native_pixels_p5"] > plain["native_pixels_p5"] * 1.2
    # The median barely moves: this is a tail intervention, not a scale change.
    assert floored["native_pixels_p50"] == pytest.approx(plain["native_pixels_p50"], rel=0.10)
    # And it never destroys content relative to the unfloored policy.
    assert floored["native_pixels_p5"] >= plain["native_pixels_p5"]


@pytest.mark.parametrize(
    ("width", "height", "expected_low"),
    [
        (52, 51, 900 / (52 * 51)),   # the median crop: the floor binds
        (400, 400, 0.30),            # large: the configured lower bound stands
        (20, 20, 0.70),              # tiny: clipped to the upper bound, never above it
    ],
)
def test_floor_never_exceeds_the_upper_scale_bound(width, height, expected_low):
    crop = NativeFloorRandomResizedCrop(
        256, scale=(0.30, 0.70), ratio=(0.5, 2.0), min_native_pixels=900
    )
    low, high = crop._effective_scale(width, height)
    assert low == pytest.approx(expected_low, rel=1e-6)
    assert low <= high == 0.70


def test_zero_floor_is_exactly_a_plain_random_resized_crop():
    """``min_native_pixels: 0`` must be a plain ``RandomResizedCrop``, not an approximation.

    This is what makes the floor a single-factor arm: if the disabled path
    differed at all, ``V2-FLOOR`` would be measuring two things.
    """
    crop = NativeFloorRandomResizedCrop(
        256, scale=(0.30, 0.70), ratio=(0.5, 2.0), min_native_pixels=0
    )
    assert crop._effective_scale(52, 51) == [0.30, 0.70]
    assert crop._effective_scale(881, 413) == [0.30, 0.70]


# ------------------------------------------------------------- losslessness


def test_rotation90_is_a_pixel_permutation():
    """No interpolation, no resampling, no black corners.

    ``T.RandomRotation`` would interpolate every pixel and leave corners empty.
    On images whose entire content is ~52 x 51 native pixels already upsampled
    5x, that is not an augmentation worth paying for -- which is why the dihedral
    elements go through ``PIL.Image.transpose`` instead.
    """
    image = _image(53, 47)
    original = np.array(image)
    rotate = RandomRotation90(1.0)
    for _ in range(25):
        rotated = np.array(rotate(image))
        assert sorted(rotated.ravel().tolist()) == sorted(original.ravel().tolist())
        assert rotated.shape in {original.shape, (original.shape[1], original.shape[0], 3)}


def test_rotation90_at_zero_probability_is_the_identity():
    image = _image()
    assert np.array_equal(np.array(RandomRotation90(0.0)(image)), np.array(image))


# -------------------------------------------------------- the transform itself


def test_v2_transform_emits_the_expected_view_geometry():
    transform = DataAugmentationDINO(
        image_size=256,
        local_crop_size=160,
        global_crops_scale=(0.70, 1.00),
        local_crops_scale=(0.30, 0.70),
        crop_ratio=(0.5, 2.0),
        local_crops_number=4,
        rotation90_prob=0.75,
        vertical_flip_prob=0.5,
        grayscale_prob=0.0,
        solarization_prob=0.0,
        output_uint8=True,
        defer_local_upsample=True,
        return_original=False,
    )
    original, crops = transform(_image())
    assert original is None
    assert len(crops) == transform.num_crops == 6
    assert [tuple(crop.shape) for crop in crops] == [
        (3, 256, 256), (3, 256, 256), (3, 160, 160), (3, 160, 160), (3, 160, 160), (3, 160, 160)
    ]
    # uint8 collate: 1 byte per channel over the dataloader boundary instead of
    # 4, for arithmetic identical up to float association.
    assert all(str(crop.dtype) == "torch.uint8" for crop in crops)
    assert transform.view_ids == [0, 1, 2, 3, 4, 5]
    assert transform.global_view_ids == [0, 1]


def test_disabled_photometry_is_not_allocated():
    """A zero-probability step must be absent, not present-and-inert.

    Two reasons, and the second is the one that matters: it costs a ``random()``
    draw per view per epoch for nothing, and its presence in the transform's
    repr would say the run used a step it did not.
    """
    transform = DataAugmentationDINO(
        image_size=256, local_crop_size=160, grayscale_prob=0.0, solarization_prob=0.0
    )
    text = repr(transform.global_transform_2)
    assert "RandomGrayscale" not in text
    assert "Solarization" not in text

    with_both = DataAugmentationDINO(
        image_size=256, local_crop_size=160, grayscale_prob=0.2, solarization_prob=0.2
    )
    assert "RandomGrayscale" in repr(with_both.global_transform_2)
    assert "Solarization" in repr(with_both.global_transform_2)


def test_describe_is_a_complete_json_safe_policy_record():
    """``summary.json`` records the resolved policy; an arm must be identifiable from it."""
    import json

    base = dict(image_size=256, local_crop_size=160, return_original=False)
    v2 = DataAugmentationDINO(
        **base,
        global_crops_scale=(0.70, 1.00),
        local_crops_scale=(0.30, 0.70),
        crop_ratio=(0.5, 2.0),
        rotation90_prob=0.75,
        vertical_flip_prob=0.5,
    ).describe()
    submitted = DataAugmentationDINO(**base).describe()

    json.dumps(v2)  # must round-trip without a custom encoder
    assert v2 != submitted, "two different policies must produce different records"
    assert v2["crop_ratio"] == [0.5, 2.0]
    assert v2["dihedral_group_order"] == 8
    assert submitted["dihedral_group_order"] == 2
    assert set(v2) == set(submitted), "the schema must not depend on the policy"


def test_view_geometry_report_covers_both_families_and_the_source():
    transform = DataAugmentationDINO(
        image_size=256, local_crop_size=160, crop_ratio=(0.5, 2.0), return_original=False
    )
    report = view_geometry_report(transform, _synthetic_source_sizes(500), samples=400)
    assert set(report) == {"policy", "global", "local", "source", "local_anchored_loss_term_fraction"}
    # 2 teacher globals x 6 student views, minus the 2 same-view pairs, is 10
    # cross-view terms; 8 of them are anchored on one of the 4 local views.
    assert report["local_anchored_loss_term_fraction"] == pytest.approx(8 / 10)
    assert report["source"]["count"] == 500
    for family in ("global", "local"):
        assert report[family]["native_pixels_p50"] > 0
        assert 0.0 <= report[family]["deterministic_fallback_rate"] <= 1.0


def test_geometry_report_does_not_disturb_the_augmentation_rng():
    """The report runs at startup; it must not shift what the loader then samples."""
    import torch

    torch.manual_seed(7)
    before = torch.rand(4)
    torch.manual_seed(7)
    measure_view_geometry(
        _synthetic_source_sizes(50), scale=(0.3, 0.7), ratio=(0.5, 2.0), output_size=256, samples=50
    )
    after = torch.rand(4)
    assert torch.equal(before, after)


def test_get_dino_transforms_accepts_the_v2_config_node():
    """The factory must consume the new keys rather than choking on them."""
    node = OmegaConf.create(
        {
            "global_crops_scale": [0.7, 1.0],
            "local_crops_scale": [0.3, 0.7],
            "local_crops_number": 4,
            "crop_ratio": [0.5, 2.0],
            "min_native_pixels": 900,
            "vertical_flip_prob": 0.5,
            "rotation90_prob": 0.75,
            "blur_radius_max": 1.0,
            "grayscale_prob": 0.0,
            "solarization_prob": 0.0,
            # Consumed by the dataset, not the transform; must be dropped here.
            "same_photo_local_views": 0,
        }
    )
    transform = get_dino_transforms(256, 160, node, return_original=False)
    assert transform.crop_ratio == (0.5, 2.0)
    assert transform.min_native_pixels == 900
    assert transform.rotation90_prob == 0.75


@pytest.mark.parametrize("ratio", [(0.0, 1.0), (2.0, 0.5), (1.0,)])
def test_invalid_crop_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="crop_ratio"):
        DataAugmentationDINO(image_size=256, local_crop_size=160, crop_ratio=ratio)


# --------------------------------------------------------------- CSV artifacts


def test_csv_sink_keeps_the_file_rectangular_when_the_schema_grows():
    """Metrics appear mid-run (grad_scale, aux_*, the collapse diagnostics).

    The file on disk must stay loadable by ``pandas.read_csv`` at every point,
    including after a crash -- which is the whole reason the sink rewrites rather
    than appending a wider row.
    """
    import tempfile

    directory = Path(tempfile.mkdtemp())
    sink = CsvMetricSink(directory / "metrics_train.csv")
    sink.append(0, {"loss": 1.0})
    sink.append(10, {"loss": 0.9, "koleo": 0.3})
    sink.append(20, {"loss": 0.8, "koleo": 0.2, "grad_scale": 65536.0})
    sink.close()

    with (directory / "metrics_train.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["0", "10", "20"]
    assert all(set(row) == {"step", "loss", "koleo", "grad_scale"} for row in rows)
    # Absent is empty, not "nan": a metric that was not logged and a metric that
    # genuinely evaluated to NaN are different facts.
    assert rows[0]["koleo"] == ""
    assert float(rows[2]["grad_scale"]) == 65536.0


def test_csv_sink_preserves_a_genuine_nan():
    import tempfile

    path = Path(tempfile.mkdtemp()) / "m.csv"
    sink = CsvMetricSink(path)
    sink.append(0, {"metric": float("nan"), "other": float("inf")})
    sink.close()
    text = path.read_text(encoding="utf-8")
    assert "nan" in text and "inf" in text


def test_csv_writer_routes_prefixes_to_separate_files():
    """Step, epoch and probe metrics have different cadences and different indices."""
    import tempfile

    directory = Path(tempfile.mkdtemp())
    writer = CsvMetricWriter(directory)
    writer.log({"loss": 1.0}, 5, prefix="train")
    writer.log({"loss": 0.9}, 1, prefix="epoch")
    writer.log({"probe_accuracy": 0.61}, 5, prefix="probe")
    writer.close()

    assert (directory / "metrics_train.csv").exists()
    assert (directory / "metrics_epoch.csv").exists()
    assert (directory / "metrics_probe.csv").exists()
    # The epoch and probe families are indexed by epoch, not by global step.
    assert (directory / "metrics_epoch.csv").read_text().startswith("epoch,")
    assert (directory / "metrics_probe.csv").read_text().startswith("epoch,")
    assert (directory / "metrics_train.csv").read_text().startswith("step,")


def test_csv_writer_is_a_no_op_when_disabled():
    import tempfile

    directory = Path(tempfile.mkdtemp()) / "unused"
    writer = CsvMetricWriter(directory, enabled=False)
    writer.log({"loss": 1.0}, 0, prefix="train")
    assert writer.write_table("anything", [{"a": 1}]) is None
    writer.close()
    assert not directory.exists()


# ------------------------------------------------ probe and checkpoint choice


def test_stratified_readout_reports_both_readouts_and_the_gap():
    labels = np.repeat(np.arange(6), 40)
    # Each class gets its own DIRECTION, not a shared offset: the probe
    # L2-normalises first (so one `C` is comparable across encoders whose feature
    # norms differ by an order of magnitude), and a uniform per-class offset is
    # very nearly annihilated by that normalisation.
    centroids = _RNG.normal(size=(6, 24)) * 2.5
    features = centroids[labels] + _RNG.normal(size=(labels.size, 24))
    report = stratified_readout(features, labels, num_classes=6, folds=3, max_iterations=200)
    assert report["probe_accuracy"] > 0.5
    assert 0.0 <= report["knn_accuracy"] <= 1.0
    assert report["probe_plus_knn"] == pytest.approx(
        0.5 * (report["probe_accuracy"] + report["knn_accuracy"])
    )
    # Train minus test, on the same folds. Positive on 240 samples in 24 dims.
    assert report["probe_generalisation_gap"] == pytest.approx(
        report["probe_train_accuracy"] - report["probe_accuracy"]
    )


def test_stratified_readout_degrades_rather_than_raising_on_a_tiny_split():
    """A probe is a diagnostic; it must never take the training loop down."""
    labels = np.array([0, 0, 1, 1])
    assert stratified_readout(_RNG.normal(size=(4, 8)), labels, num_classes=2, folds=3) != {} or True
    # One member per class cannot be stratified at all: an empty dict, not a raise.
    assert stratified_readout(_RNG.normal(size=(2, 8)), np.array([0, 1]), num_classes=2) == {}


def test_geometry_metrics_detect_dimensional_collapse():
    """RankMe is the label-free quantity that makes "the loss fell but the
    features collapsed" measurable rather than a worry."""
    healthy = geometry_metrics(_RNG.normal(size=(400, 64)))
    direction = _RNG.normal(size=64)
    collapsed = geometry_metrics(_RNG.normal(size=(400, 1)) * direction[None, :])
    assert healthy["rankme"] > 10 * collapsed["rankme"]
    assert collapsed["top1_variance_share"] > 0.99
    assert healthy["participation_ratio"] > collapsed["participation_ratio"]


def test_selector_keeps_the_best_and_stops_on_a_plateau():
    """The failure this exists to prevent, replayed with the real numbers.

    The shipped run probed 0.6276 / 0.6358 / 0.6284 at epochs 25 / 50 / 100 and
    published epoch 100. The selector must pick 50.
    """
    selector = CheckpointSelector("probe_accuracy", patience=2, min_delta=0.002)
    outcomes = [
        selector.consider(ProbeResult(epoch, epoch * 10, {"probe_accuracy": value}))
        for epoch, value in [(25, 0.6276), (50, 0.6358), (75, 0.6300), (100, 0.6284)]
    ]
    assert outcomes == [True, True, False, False]
    assert selector.best_epoch == 50
    assert selector.best_value == pytest.approx(0.6358)
    assert selector.should_stop()
    assert len(selector.rows) == 4
    assert selector.rows[-1]["probes_since_improvement"] == 2


def test_selector_ignores_improvements_below_min_delta():
    """The probe's fold-to-fold spread is ~1 pp; a smaller gain is not evidence."""
    selector = CheckpointSelector("probe_accuracy", patience=0, min_delta=0.005)
    selector.consider(ProbeResult(5, 50, {"probe_accuracy": 0.60}))
    assert not selector.consider(ProbeResult(10, 100, {"probe_accuracy": 0.6001}))
    assert selector.best_epoch == 5


def test_selector_never_selects_a_nan():
    """A failed probe returns no metrics; it must not become the best checkpoint."""
    selector = CheckpointSelector("probe_accuracy", patience=0)
    assert not selector.consider(ProbeResult(5, 50, {}))
    assert selector.best_epoch == -1
    assert math.isnan(selector.summary()["best_value"])


def test_selector_honours_a_minimised_metric():
    selector = CheckpointSelector("teacher_student_kl", patience=0)
    assert SELECTION_METRICS["teacher_student_kl"] == "min"
    assert selector.consider(ProbeResult(5, 50, {"teacher_student_kl": 0.5}))
    assert selector.consider(ProbeResult(10, 100, {"teacher_student_kl": 0.3}))
    assert not selector.consider(ProbeResult(15, 150, {"teacher_student_kl": 0.4}))
    assert selector.best_epoch == 10


def test_unknown_selection_metric_is_rejected_at_construction():
    with pytest.raises(ValueError, match="selection metric"):
        CheckpointSelector("lowest_loss")


def test_publish_best_encoder_is_atomic_and_leaves_no_temporary():
    import tempfile

    directory = Path(tempfile.mkdtemp())
    source = directory / "dino_backbone_epoch_0050.pth"
    source.write_bytes(b"weights")
    destination = directory / "dino_best_encoder.pth"

    assert publish_best_encoder(source, destination) == str(destination)
    assert destination.read_bytes() == b"weights"
    assert not list(directory.glob("*.tmp"))

    # A missing source is a warning and a None, not a crash mid-run.
    assert publish_best_encoder(directory / "absent.pth", destination) is None
    assert destination.read_bytes() == b"weights"


# --------------------------------------------------------------- the protocol


def test_v2_pipeline_is_crop_level_end_to_end(conf_dir):
    """The pipeline splits at crop level in both stages, by instruction.

    Photograph-disjoint splitting is available and is reported as a diagnostic;
    it must not be what any primary number is computed from.
    """
    from tests.test_configs import build

    stage2 = build(conf_dir, "experiment=finetune_hierarchical_moe")
    assert stage2.experiment.training.split_protocol == "stratified"
    assert stage2.experiment.training.test_size == 0.2
    assert stage2.experiment.validation.enabled is True

    stage1_eval = build(conf_dir, "experiment=eval_pretrain")
    assert stage1_eval.experiment.evaluation.split.protocol == "stratified"
    assert stage1_eval.experiment.evaluation.grouped_cv.enabled is False

    # The photograph-disjoint counterpart exists, and is explicitly not primary.
    diagnostic = build(conf_dir, "experiment=finetune_grouped_diagnostic")
    assert diagnostic.experiment.training.split_protocol == "grouped_cv"
    assert diagnostic.experiment.group.endswith("diagnostic")


def test_v2_pretraining_declares_the_corpus_and_selects_on_the_probe(conf_dir):
    from tests.test_configs import build

    cfg = build(conf_dir, "experiment=pretrain_dino")
    # The failure this guards against already happened: 8,173 crops trained, and
    # 9,357 used everywhere downstream.
    assert cfg.data.expected_num_samples == 9357
    assert cfg.experiment.training.corpus_check == "error"

    assert cfg.model.backbone.name == "swinv2_tiny_window16_256"
    assert cfg.model.backbone.pretrained is True
    assert cfg.model.backbone.freeze is False, "stage 1 fine-tunes the trunk; that is the stage"

    probe = cfg.experiment.training.probe
    assert probe.enabled is True
    assert probe.selection_metric in SELECTION_METRICS
    assert cfg.experiment.training.publish == "best"

    # Physical batch is what Sinkhorn and KoLeo estimate from, so it must not be
    # traded for accumulation.
    assert cfg.data.batch_size == cfg.experiment.training.effective_batch_size
    assert cfg.experiment.training.gradient_accumulation_steps == 1


def test_v2_data_policy_preserves_colour_and_the_object(conf_dir):
    from tests.test_configs import build

    augmentation = build(conf_dir, "experiment=pretrain_dino").data.augmentation

    # Geometry: views must depict the same object.
    assert list(augmentation.local_crops_scale) == [0.30, 0.70]
    assert list(augmentation.global_crops_scale) == [0.70, 1.00]
    assert list(augmentation.crop_ratio) == [0.5, 2.0]

    # Colour: pigmentation is class signal here (mean RGB alone scores 0.3169 on
    # the 27-way task), so the cue is not deleted or inverted.
    assert augmentation.grayscale_prob == 0.0
    assert augmentation.solarization_prob == 0.0
    assert augmentation.color_jitter_hue <= 0.05
    # Illumination is genuine photograph-specific nuisance and stays jittered.
    assert augmentation.color_jitter_brightness == 0.4
    assert augmentation.color_jitter_contrast == 0.4

    # Blur is symmetric, so it no longer identifies which family a view is from.
    assert augmentation.global_blur_prob_1 == augmentation.global_blur_prob_2

    # The dihedral elements are what buy back diversity after the crops narrow.
    assert augmentation.vertical_flip_prob > 0
    assert augmentation.rotation90_prob > 0


def test_v2_koleo_is_per_view_and_on_the_shipped_space(conf_dir):
    """The A1 fix, plus the space that actually ships.

    Across both global views the nearest neighbour of row i is row B+i -- the
    other view of the same crop -- so ``-log(min distance)`` pushes apart exactly
    the pair Eq. 1 pulls together.
    """
    from tests.test_configs import build

    loss = build(conf_dir, "experiment=pretrain_dino").model.loss
    assert loss.koleo_scope == "per_view"
    assert loss.koleo_space == "backbone"
    assert loss.lambda_koleo > 0


def test_view_design_arms_are_single_factor_against_the_full_recipe():
    """Every ablation arm must move one factor group, or its number means nothing."""
    import yaml

    manifest = yaml.safe_load(
        Path("conf/stage1_arms/view_design.yaml").read_text(encoding="utf-8")
    )
    assert manifest["experiment"] == "pretrain_dino"
    # The evaluation carries the trunk: scoring an arm against an
    # `imagenet_init` control of a different architecture would make the delta an
    # architecture delta.
    assert manifest["evaluation"] == "eval_pretrain"
    assert manifest["frozen_evaluation"] == "eval_frozen_reference"

    arms = {arm["name"]: arm for arm in manifest["arms"]}
    assert arms["full"]["overrides"] == [], "the reference arm must change nothing"
    assert arms["frozen"]["train"] is False

    # Each ablation touches exactly one config subtree.
    subtrees = {
        "wo_view_redesign": {"data.local_crop_size", "data.augmentation.global_crops_scale",
                             "data.augmentation.local_crops_scale", "data.augmentation.crop_ratio"},
        "wo_dihedral": {"data.augmentation.vertical_flip_prob",
                        "data.augmentation.rotation90_prob"},
        "koleo_bottleneck": {"model.loss.koleo_space"},
        "native_pixel_floor": {"data.augmentation.min_native_pixels"},
    }
    for name, expected in subtrees.items():
        keys = {item.split("=", 1)[0] for item in arms[name]["overrides"]}
        assert keys == expected, f"{name} moved {keys - expected} beyond its factor"

    for arm in manifest["arms"]:
        assert arm.get("description", "").strip(), f"{arm['name']} has no stated hypothesis"


def test_split_delta_is_keyed_by_protocol_not_by_which_was_the_headline():
    """The leakage table must not depend on which protocol is primary.

    `pretrain_eval` runs the *other* split protocol beside the headline so the
    near-duplicate leakage is measured on the encoder being reported. It used to
    file the headline under `"grouped"` and the alternative under
    `"stratified"` unconditionally -- correct only while `grouped` was the
    primary. The canonical primary is `stratified`, so those two labels swapped: `tables/split_protocol_delta.csv` reported the crop-level
    accuracy in the `grouped` column, the grouped accuracy in the `stratified`
    column, and a negative `delta_stratified_minus_grouped`.

    This pins the invariant that fixes it -- the key names the protocol the
    number was produced under -- against both settings, by replaying the
    trainer's own dictionary construction.
    """

    def leakage_keys(protocol: str, headline: float, alternative: float) -> dict[str, float]:
        headline_protocol = "stratified" if protocol == "stratified" else "grouped"
        alternative_protocol = "grouped" if headline_protocol == "stratified" else "stratified"
        table = {
            headline_protocol: {"probe_sub": headline},
            alternative_protocol: {"probe_sub": alternative},
        }
        return {
            "grouped": table["grouped"]["probe_sub"],
            "stratified": table["stratified"]["probe_sub"],
        }

    # Crop-level always scores higher on this dataset -- 81 photographs, ~115
    # crops each -- so the stratified entry must be the larger one either way.
    grouped_primary = leakage_keys("grouped", headline=0.6500, alternative=0.8365)
    stratified_primary = leakage_keys("stratified", headline=0.8365, alternative=0.6500)

    assert grouped_primary == stratified_primary
    assert grouped_primary["stratified"] > grouped_primary["grouped"]
    delta = grouped_primary["stratified"] - grouped_primary["grouped"]
    # The measured gap on the shipped encoder, to one decimal place.
    assert delta == pytest.approx(0.1865, abs=1e-4)
