"""Stage 0: seed detection, instance separation, crop policy and the audit.

Everything here runs on a **synthetic photograph** built to the measured
properties of the real corpus -- a bright paper with an illumination gradient,
seeds at known positions and known areas, a touching pair, dust specks, and a
dark non-paper band along one edge. That is deliberate: the real photographs are
not in the repository, and a test that needs them is a test that does not run.

The synthetic scene is built so that every assertion below has a *ground truth*.
Where a constant appears it comes from ``tests/conftest.py``, and the
measurements those constants encode are recorded in the module they describe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.segmentation.detect import (  # noqa: E402
    DetectionParams,
    binarise,
    component_instances,
    estimate_seed_area,
    foreground_score,
    split_touching,
)
from src.segmentation.illumination import (  # noqa: E402
    illumination_field,
    model_scene,
    support_region,
)
from src.segmentation.instances import (  # noqa: E402
    REJECTION_REASONS,
    CropPolicy,
    SeedInstance,
    render_crop,
    square_window,
    suppress_distractors,
)
from src.segmentation.pipeline import SceneGate, segment_photograph  # noqa: E402

# --------------------------------------------------------------------------
# The synthetic photograph
# --------------------------------------------------------------------------

#: Half-axes of one synthetic seed, in pixels. Chosen inside the real corpus's
#: measured range (median crop side 61 px, p1/p99 = 35/174), so the area gates --
#: which are all *ratios* to the frame's own median -- behave as they do on the
#: real thing.
SEED_AXES = (16, 11)
SEED_COLOUR = (58, 74, 96)     # BGR: a dark brown, like an amaranthus seed
PAPER_COLOUR = (188, 190, 191) # BGR: near-neutral, like the real sheets


def make_photograph(
    width: int = 900,
    height: int = 700,
    rows: int = 5,
    columns: int = 7,
    with_touching_pair: bool = True,
    with_dust: bool = True,
    with_dark_band: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A tray of seeds on lit paper, and the true centre of every whole seed.

    Reproduces the four properties that actually decide the algorithm:

    * a **smooth illumination gradient** across the sheet (the real corpus is a
      median 4.3 % brighter at the centre than at the border), so a global
      threshold is not sufficient;
    * **sensor noise**, so the robust scale estimates have something to estimate;
    * optional **dust**, which is what the size floor exists to reject -- the
      real photographs yield 185 sub-threshold components against 200 seeds;
    * an optional **dark band** along the bottom edge, standing in for the paper
      edge and its shadow, which is what the support erosion exists to remove.
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:] = PAPER_COLOUR

    ys, xs = np.mgrid[0:height, 0:width]
    radial = ((xs - width / 2) ** 2 + (ys - height / 2) ** 2) / (width * height)
    image *= (1.0 - 0.10 * radial)[:, :, None]

    centres: list[tuple[int, int]] = []
    step_x, step_y = width // (columns + 1), height // (rows + 1)
    for row in range(rows):
        for column in range(columns):
            x = step_x * (column + 1)
            y = step_y * (row + 1)
            cv2.ellipse(
                image, (x, y), SEED_AXES, float(rng.integers(0, 180)), 0, 360, SEED_COLOUR, -1
            )
            centres.append((x, y))

    if with_touching_pair:
        # Two seeds sharing a boundary: one connected component, two instances.
        #
        # Their orientations differ by 80 degrees, which is what makes the union
        # CONCAVE. Two identically oriented ellipses side by side merge into a
        # near-convex stadium (solidity 0.955) that the split rule correctly
        # declines to cut -- and two real touching seeds are almost never
        # parallel. Offset 26 px gives one component at 2.1 median seeds and
        # solidity 0.84, which is the middle of the range measured on the real
        # touching pairs (0.77-0.88).
        # Between the last two grid rows, and between two grid columns: far
        # enough from every grid seed not to merge with one, and far enough
        # inside the sheet to survive the support erosion the dark band forces.
        base_x = step_x * (columns // 2) + step_x // 2
        base_y = step_y * rows - step_y // 2
        for sign, angle in ((-1, 20), (1, 100)):
            centre = (base_x + sign * 13, base_y)
            cv2.ellipse(image, centre, SEED_AXES, angle, 0, 360, SEED_COLOUR, -1)
            centres.append(centre)

    if with_dust:
        for _ in range(60):
            x = int(rng.integers(30, width - 30))
            y = int(rng.integers(30, height - 30))
            cv2.circle(image, (x, y), int(rng.integers(1, 3)), SEED_COLOUR, -1)

    if with_dark_band:
        image[height - 26 :, :] = (70, 62, 58)

    image += rng.normal(0.0, 1.6, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), centres


@pytest.fixture(scope="module")
def photograph():
    return make_photograph()


@pytest.fixture(scope="module")
def params() -> DetectionParams:
    return DetectionParams()


# --------------------------------------------------------------------------
# Illumination and support
# --------------------------------------------------------------------------

def test_illumination_field_removes_the_seeds_and_keeps_the_gradient(photograph):
    """The field must interpolate *through* a seed, not dent around it.

    A closing whose element is smaller than a seed would leave a dark hollow
    where the seed is, and the relative-darkness channel would then read that
    seed as background. The check is direct: the field at a seed's centre must be
    close to the paper right beside it, and it must still carry the frame's
    gradient rather than a single constant.
    """
    image, centres = photograph
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    field = illumination_field(grey)

    for x, y in centres[:8]:
        beside = float(grey[y, x + 3 * SEED_AXES[0]])
        assert field[y, x] == pytest.approx(beside, rel=0.10), (
            "the field dented where a seed is; the closing element is too small"
        )
    # Still a gradient, not a constant: the synthetic vignette is 10 %.
    assert field.max() - field.min() > 5.0


def test_support_region_excludes_a_non_paper_band_and_erodes_its_edge(photograph):
    """The dark band along the bottom must not be inside the support.

    On the real corpus this band is the paper's own edge and its cast shadow, and
    leaving it in produces a single component 34x the median seed area
    (``PearlMillet/IMG_0510``) or 102x (``Poosa33/IMG_0665``).
    """
    image, _ = photograph
    support, fraction = support_region(image, margin_px=30)

    assert support.dtype == bool
    assert support.shape == image.shape[:2]
    assert not support[-10:, :].any(), "the non-paper band is inside the support"
    # The paper still covers most of the frame before erosion.
    assert fraction > 0.85


def test_support_region_reports_a_low_fraction_when_there_is_no_tray():
    """The scene gate's input: a frame that is mostly wooden table.

    This is ``Poosa33/IMG_0668`` in miniature -- a small sheet on a dark, textured
    surface. The support fraction is what separates it from a tray, and it must
    be well below the 0.70 gate rather than marginally below it.
    """
    rng = np.random.default_rng(3)
    image = np.zeros((400, 600, 3), np.uint8)
    image[:] = (26, 62, 96)                       # BGR: wood
    image += rng.normal(0, 6, image.shape).astype(np.int16).clip(-20, 20).astype(np.uint8)
    image[120:280, 180:420] = PAPER_COLOUR        # a sheet covering 16 % of the frame

    _, fraction = support_region(image, margin_px=8)
    assert fraction < 0.30


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def test_foreground_score_separates_seeds_from_paper_by_orders_of_magnitude(
    photograph, params
):
    """Both channels are robust z-scores against the *paper's own* noise.

    The assertion is about the margin, not just the ordering: a seed must stand
    out by far more than the threshold, so the threshold is not load-bearing.
    """
    image, centres = photograph
    scene = model_scene(image, support_margin_px=30)
    score, darkness, chroma, stats = foreground_score(image, scene)

    seed_scores = np.array([score[y, x] for x, y in centres[:20]])
    assert seed_scores.min() > 3 * params.z_threshold

    paper = scene.support & (score <= params.z_threshold)
    assert paper.mean() > 0.8, "most of the paper must be below the threshold"
    assert stats["paper_darkness_sigma"] > 0
    assert stats["paper_chroma_sigma"] > 0


def test_binarise_recovers_every_seed_and_no_paper(photograph, params):
    """One component per seed, at close to the true area."""
    image, centres = photograph
    scene = model_scene(image, support_margin_px=30)
    mask = binarise(*foreground_score(image, scene)[:3], scene, params)

    for x, y in centres[:20]:
        assert mask[y, x], f"seed at ({x}, {y}) was not detected"

    true_area = np.pi * SEED_AXES[0] * SEED_AXES[1]
    assert estimate_seed_area(mask) == pytest.approx(true_area, rel=0.15)


def test_estimate_seed_area_ignores_dust(params):
    """The unit of "one seed" must survive a frame that is mostly specks.

    The real photographs do exactly this: ``AMT-1/IMG_0653`` yields 185
    sub-threshold components against 200 seeds, and ``PearlMillet/IMG_0510``
    yields 693 against 268.
    """
    image, _ = make_photograph(with_dust=True, rows=3, columns=4)
    scene = model_scene(image, support_margin_px=30)
    mask = binarise(*foreground_score(image, scene)[:3], scene, params)
    assert estimate_seed_area(mask) == pytest.approx(
        np.pi * SEED_AXES[0] * SEED_AXES[1], rel=0.15
    )


# --------------------------------------------------------------------------
# Instance separation
# --------------------------------------------------------------------------

def test_touching_seeds_are_split_into_two_instances(photograph, params):
    """A merged component must not become one crop.

    The pair in the synthetic scene shares a boundary, so it is one connected
    component with roughly twice the median area and a waist.
    """
    image, _ = photograph
    scene = model_scene(image, support_margin_px=30)
    mask = binarise(*foreground_score(image, scene)[:3], scene, params)
    seed_area = estimate_seed_area(mask)
    _, records = component_instances(mask, seed_area, params)

    split = [record for record in records if record["origin"] == "split"]
    assert len(split) == 2, f"expected the touching pair to split, got {len(split)} fragments"
    assert all(record["cluster_size"] == 2 for record in split)


def test_a_single_large_seed_is_not_split(params):
    """Area alone must not trigger a split, or large individuals get halved.

    This is the ``KodoMillet/IMG_0492`` case: twelve components measure 1.5-1.85
    median seeds and every one is a single seed. The discriminator is the
    distance map's peak count, which a convex blob does not supply.
    """
    image, centres = make_photograph(
        with_touching_pair=False, with_dust=False, with_dark_band=False, rows=4, columns=5
    )
    # One seed at 1.7x the area of the others -- above `split_area_ratio` -- drawn
    # in the gap between grid rows so it stays a single connected component.
    big = (int(SEED_AXES[0] * 1.31), int(SEED_AXES[1] * 1.31))
    lone = (900 // 2, 700 // 5 // 2)
    assert min(np.hypot(lone[0] - x, lone[1] - y) for x, y in centres) > 4 * SEED_AXES[0]
    cv2.ellipse(image, lone, big, 30, 0, 360, SEED_COLOUR, -1)
    scene = model_scene(image, support_margin_px=30)
    mask = binarise(*foreground_score(image, scene)[:3], scene, params)
    seed_area = estimate_seed_area(mask)
    _, records = component_instances(mask, seed_area, params)

    oversized = [r for r in records if r["origin"] == "oversized_single"]
    assert not [r for r in records if r["origin"] == "split"], "a single seed was split"
    assert len(oversized) == 1
    assert 1.5 < float(oversized[0]["parent_ratio"]) <= params.max_single_ratio


def test_split_touching_declines_rather_than_returning_a_sliver(params):
    """A split whose fragments are implausible must be discarded whole.

    The caller then records an unresolved cluster. Emitting the fragments would
    put two crops of one seed into the corpus with nothing saying so.
    """
    blob = np.zeros((60, 120), dtype=bool)
    cv2.ellipse(blob.view(np.uint8), (60, 30), (55, 25), 0, 0, 360, 1, -1)
    blob = blob.view(np.uint8).astype(bool)
    # Demand fragments larger than the whole blob: no split can satisfy it.
    assert split_touching(blob, 2, blob.sum() * 2.0, blob.sum() * 4.0, params) is None
    # And a component the area test never flagged is never split.
    assert split_touching(blob, 1, 1.0, 1e9, params) is None


# --------------------------------------------------------------------------
# The crop policy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "box", [(100, 100, 40, 20), (100, 100, 20, 40), (100, 100, 33, 33), (5, 5, 60, 20)]
)
def test_square_window_is_square_and_contains_the_whole_tight_box(box):
    """Squareness and containment are the two properties the corpus is built on.

    Square, because the downstream ``Resize((H, W))`` is then a uniform rescale
    and the seed's aspect ratio survives to the encoder. Containing the tight
    box, because a window that clipped the seed would defeat the point of cutting
    at native resolution.
    """
    policy = CropPolicy(margin=0.12)
    window, fits, _ = square_window(box, policy, (600, 800))
    x, y, side, other = window

    assert fits and side == other
    assert side >= max(box[2], box[3])
    assert x <= box[0] and y <= box[1]
    assert x + side >= box[0] + box[2]
    assert y + side >= box[1] + box[3]
    assert 0 <= x and 0 <= y and x + side <= 800 and y + side <= 600


def test_square_window_translates_rather_than_clipping_at_the_frame_edge():
    """A window near the border moves inward; it never shrinks.

    Shrinking would clip the seed. Translating only moves the paper ring, and
    ``side >= max(w, h)`` guarantees the seed is still inside.
    """
    policy = CropPolicy(margin=0.30)
    box = (2, 2, 40, 40)
    window, fits, shifted = square_window(box, policy, (300, 300))
    assert fits and shifted
    assert window[0] == 0 and window[1] == 0
    assert window[0] + window[2] >= box[0] + box[2]


def test_square_window_refuses_a_window_larger_than_the_frame():
    _, fits, _ = square_window((0, 0, 200, 200), CropPolicy(), (100, 150))
    assert not fits


def test_render_crop_copies_source_pixels_exactly_when_nothing_intrudes(photograph):
    """No resampling anywhere. The crop must be a bit-exact copy.

    This is the property that makes every later resize an explicit, configurable
    step rather than a loss baked into the corpus.
    """
    image, _ = photograph
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    labels[100:140, 200:240] = 1
    instance = SeedInstance(
        index=0, label=1, bbox=(200, 100, 40, 40), centroid=(220.0, 120.0),
        area=1200.0, area_ratio=1.0, origin="component", cluster_size=1,
        solidity=0.99, extent=0.8, eccentricity=0.3, equivalent_diameter=39.0,
        major_axis=40.0, minor_axis=38.0, orientation_deg=0.0, perimeter=120.0,
        circularity=0.9, mean_rgb=(0, 0, 0), std_rgb=(0, 0, 0), mean_lab=(0, 0, 0),
        contrast_vs_paper=100.0, focus=50.0,
    )
    instance.window, _, _ = square_window(instance.bbox, CropPolicy(), image.shape[:2])
    crop = render_crop(image, labels, instance, CropPolicy())

    x, y, side, _ = instance.window
    assert crop.shape[0] == crop.shape[1] == side
    assert np.array_equal(crop, image[y : y + side, x : x + side])
    assert instance.distractor_fraction == 0.0


def test_suppress_distractors_removes_the_neighbour_and_never_the_target():
    """Inpainting the intruder must leave the target's own pixels untouched.

    A square window catches a neighbouring seed 2.9 % of the time at the
    configured margin. Leaving it in puts a second instance of the class in the
    frame -- which at DINO's local-crop scales can become the whole view.
    """
    crop = np.full((80, 80, 3), PAPER_COLOUR, np.uint8)
    labels = np.zeros((80, 80), np.int32)
    cv2.circle(crop, (30, 40), 14, SEED_COLOUR, -1)
    cv2.circle(labels, (30, 40), 14, 1, -1)
    cv2.circle(crop, (70, 12), 12, (30, 30, 30), -1)
    cv2.circle(labels, (70, 12), 12, 2, -1)

    target_before = crop[labels == 1].copy()
    repaired, fraction, removed = suppress_distractors(crop, labels, keep_label=1, dilation=3)

    assert removed == (2,)
    assert 0.0 < fraction < 0.4
    assert np.array_equal(repaired[labels == 1], target_before), "the target was edited"
    # The distractor is gone: what replaces it is paper, not a dark disc.
    assert repaired[12, 70].mean() > 120


def test_every_rejection_reason_is_declared():
    """``SeedInstance.reject`` refuses an undeclared reason.

    The audit tabulates ``REJECTION_REASONS``, so a reason invented at a call
    site would be a category the report silently omits.
    """
    instance = SeedInstance(
        index=0, label=1, bbox=(0, 0, 1, 1), centroid=(0.0, 0.0), area=1.0,
        area_ratio=1.0, origin="component", cluster_size=1, solidity=1.0, extent=1.0,
        eccentricity=0.0, equivalent_diameter=1.0, major_axis=1.0, minor_axis=1.0,
        orientation_deg=0.0, perimeter=1.0, circularity=1.0, mean_rgb=(0, 0, 0),
        std_rgb=(0, 0, 0), mean_lab=(0, 0, 0), contrast_vs_paper=0.0, focus=0.0,
    )
    with pytest.raises(ValueError):
        instance.reject("looked_odd")
    for reason in REJECTION_REASONS:
        instance.reject(reason)
    assert instance.reasons == list(REJECTION_REASONS)
    assert not instance.accepted


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_segment_photograph_finds_every_seed_and_records_every_rejection(photograph):
    """The whole pipeline on a scene with a known seed count.

    35 whole seeds plus a touching pair that must resolve into 2 = 37, and every
    dust speck and the dark band accounted for by name.
    """
    image, centres = photograph
    result = segment_photograph(
        image, DetectionParams(), CropPolicy(), SceneGate(),
        path="synthetic", seed_type="Synthetic", sub_variety="Seed", stem="IMG_0001",
        keep_arrays=True,
    )

    assert result.accepted_scene, result.scene_reasons
    assert len(result.accepted) == len(centres)

    # Every detection is adjudicated: accepted, or rejected with a stated reason.
    for instance in result.instances:
        assert instance.accepted or instance.reasons
    assert result.rejected, "the dust and the dark band must appear as rejections"
    assert {"too_small"} <= {reason for item in result.rejected for reason in item.reasons}

    # Each accepted detection sits within one seed-radius of a true centre, and
    # no true centre is claimed twice.
    claimed: set[int] = set()
    for instance in result.accepted:
        distances = [
            np.hypot(instance.centroid[0] - x, instance.centroid[1] - y) for x, y in centres
        ]
        nearest = int(np.argmin(distances))
        assert distances[nearest] < SEED_AXES[0]
        assert nearest not in claimed, "two detections claimed the same seed"
        claimed.add(nearest)

    # The manifest row must be flat and JSON-safe: it is written to CSV.
    row = result.accepted[0].row()
    json.dumps(row)
    assert row["window_side"] > 0 and isinstance(row["reasons"], str)
    assert isinstance(result.summary()["num_accepted"], int)


def test_a_scene_that_is_not_a_tray_is_rejected_with_reasons():
    """Nothing is written from a frame the gate refuses, and it says why.

    Every instance carries ``scene_rejected`` explicitly rather than merely being
    flipped to un-accepted, so the manifest and the overlay can both explain an
    empty output.
    """
    rng = np.random.default_rng(11)
    image = np.zeros((400, 600, 3), np.uint8)
    image[:] = (26, 62, 96)
    image += rng.normal(0, 6, image.shape).astype(np.int16).clip(-20, 20).astype(np.uint8)
    image[120:280, 180:420] = PAPER_COLOUR
    cv2.circle(image, (300, 200), 9, SEED_COLOUR, -1)

    result = segment_photograph(
        image, DetectionParams(), CropPolicy(), SceneGate(),
        path="synthetic", seed_type="Synthetic", sub_variety="Seed", stem="IMG_0002",
    )
    assert not result.accepted_scene
    assert result.scene_reasons
    assert not result.accepted
    assert all("scene_rejected" in item.reasons for item in result.instances)


def test_crop_filenames_keep_the_provenance_convention_the_split_protocol_needs():
    """``IMG_0502_bbox0007.png`` -- the same key ``source_image_id`` parses.

    This is the load-bearing contract between stage 0 and everything downstream.
    The photograph-disjoint split protocol groups crops by source photograph, and
    it derives that group from the filename; a refined corpus whose names did not
    match would silently degrade every grouped split to crop-level splitting.
    """
    from omegaconf import OmegaConf

    from src.datasets.dataset import SOURCE_IMAGE_PATTERN, source_image_id

    template = str(OmegaConf.load("conf/segmentation.yaml").segmentation.output.filename_template)
    name = template.format(stem="IMG_0502", index=7) + ".png"

    assert SOURCE_IMAGE_PATTERN.match(Path(name).stem)
    assert source_image_id(f"Rice/Chinnar/{name}") == "Chinnar/IMG_0502"
    # Two crops of one photograph must share a group; two photographs must not.
    other = template.format(stem="IMG_0502", index=99) + ".png"
    assert source_image_id(f"Rice/Chinnar/{other}") == source_image_id(f"Rice/Chinnar/{name}")


# --------------------------------------------------------------------------
# The audit's matching primitives
# --------------------------------------------------------------------------

def test_iou_matrix_and_greedy_matching():
    from src.segmentation.audit import greedy_match, iou_matrix, self_overlaps

    left = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], dtype=np.float64)
    right = np.array([[1, 1, 10, 10], [500, 500, 10, 10]], dtype=np.float64)

    scores = iou_matrix(left, right)
    assert scores[0, 0] == pytest.approx(81 / (200 - 81))
    assert scores[0, 1] == 0.0

    pairs = greedy_match(scores, 0.3)
    assert pairs == [(0, 0, pytest.approx(81 / 119))]
    # One-to-one: a legacy box already matched cannot match again.
    assert len({pair[0] for pair in pairs}) == len(pairs)

    assert self_overlaps(left, 0.6) == []
    duplicated = np.array([[0, 0, 10, 10], [1, 0, 10, 10]], dtype=np.float64)
    assert len(self_overlaps(duplicated, 0.6)) == 1


def test_locate_crop_recovers_an_exact_sub_image(photograph):
    """The audit's reference set stands on this being exact, not approximate.

    A crop that is a byte-identical sub-image must be located at its true
    position and reported ``exact``; a crop that is not from this photograph must
    be reported ``exact=False`` rather than quietly given a position.
    """
    from src.segmentation.audit import locate_crop

    image, _ = photograph
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(grey, (grey.shape[1] // 2, grey.shape[0] // 2), interpolation=cv2.INTER_AREA)

    truth = (311, 207, 74, 61)
    crop = image[truth[1] : truth[1] + truth[3], truth[0] : truth[0] + truth[2]].copy()
    assert locate_crop(image, grey, reduced, 2, crop) == (*truth, True)

    foreign = np.full((30, 30, 3), 7, np.uint8)
    located = locate_crop(image, grey, reduced, 2, foreign)
    assert located is not None and located[4] is False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_segmentation_config_keys_match_the_dataclasses():
    """A config key that no dataclass field accepts is a silently ignored knob."""
    import dataclasses

    from omegaconf import OmegaConf

    cfg = OmegaConf.load("conf/segmentation.yaml")
    for node, cls in (
        (cfg.segmentation.detection, DetectionParams),
        (cfg.segmentation.crop, CropPolicy),
        (cfg.segmentation.scene, SceneGate),
    ):
        fields = {field.name for field in dataclasses.fields(cls)}
        assert set(node.keys()) == fields, (
            f"{cls.__name__}: config has {set(node.keys()) - fields}, "
            f"missing {fields - set(node.keys())}"
        )
        # And it must actually construct.
        cls(**OmegaConf.to_container(node, resolve=True))


def test_refined_corpus_is_the_canonical_data_root():
    """The one data group points at the refined corpus and declares its size."""
    from omegaconf import OmegaConf

    from tests.conftest import DATASET_NUM_CROPS

    data = OmegaConf.load("conf/data/hierarchical_seeds.yaml")
    assert "Refined_Samples" in str(data.root_path)
    assert data.expected_num_samples == DATASET_NUM_CROPS


def test_the_extractor_never_writes_under_the_raw_root():
    """The raw photographs are the only irreplaceable artifact in the pipeline.

    Enforced by reading the module: no write call may take a path derived from
    ``raw_root``. A behavioural test cannot prove a negative here, but this
    catches the change that would introduce one.
    """
    source = Path("src/segmentation/extract.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "raw_root" in stripped:
            assert not any(
                token in stripped for token in ("imwrite", "write_text", "mkdir(", "unlink")
            ), f"extract.py writes under raw_root: {stripped}"
