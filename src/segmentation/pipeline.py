"""One photograph in, a list of :class:`SeedInstance` out.

This module holds the *decisions*: which detections become files, which are
recorded as rejected and why, and when a whole photograph is not a tray of seeds
at all. :mod:`src.segmentation.detect` finds candidates; nothing there knows what
a good one looks like.

The scene gate
--------------

Three of the 99 photographs in ``RAW_Samples`` are not trays: a seed packet
standing on a wooden table, a near-empty sheet, and a labelled ziplock bag held
up to the camera. Cropping them would inject wood grain, printed text and a
photograph of a burger into a seed corpus.

They are not identified by filename. They are identified by two measurements
that a genuine tray passes with a wide margin:

* **support fraction** -- the share of the frame that is one bright, neutral
  sheet. The 96 tray photographs measure 0.76-1.00 (median 1.00); the three
  others measure 0.58, 0.63 and 0.65.
* **seed population** -- how many plausible detections survive the per-instance
  gate. A tray yields 31-390; the near-empty sheet yields 2.

Both thresholds are config keys and both are *reported* for every photograph, so
a scene that was excluded says why and a scene that was kept says by how much.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from src.segmentation.detect import (
    DetectionParams,
    binarise,
    component_instances,
    estimate_seed_area,
    foreground_score,
)
from src.segmentation.illumination import model_scene
from src.segmentation.instances import (
    CropPolicy,
    SeedInstance,
    measure,
    square_window,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SceneGate:
    """When a photograph is not a tray of seeds.

    Attributes:
        min_support_fraction: Smallest share of the frame the paper may cover.
            0.70 sits between the 0.76 of the worst genuine tray
            (``Kullakar/IMG_0711``, shot at an angle) and the 0.65 of the ziplock
            bag.
        min_instances: Fewest accepted detections a tray may yield. The
            near-empty ``Poosa33/IMG_0667`` yields 2.
        max_reject_fraction: Largest share of detections that may be rejected
            before the scene itself is suspect. A frame where most candidates
            fail the shape and size tests is a frame where the detector is
            responding to something other than seeds.
    """

    min_support_fraction: float = 0.70
    min_instances: int = 20
    max_reject_fraction: float = 0.60


@dataclass(slots=True)
class PhotographResult:
    """Everything one photograph produced.

    ``instances`` holds **every** detection, accepted and rejected alike, in
    detection order. The rejected ones keep their descriptors and their reasons;
    nothing is filtered out of this list, because a count of what was thrown
    away is the only way to tell a clean photograph from a badly segmented one.
    """

    path: str
    seed_type: str
    sub_variety: str
    stem: str
    width: int
    height: int
    seed_area: float
    instances: list[SeedInstance]
    scene: dict[str, object]
    accepted_scene: bool = True
    scene_reasons: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None
    mask: np.ndarray | None = None

    @property
    def accepted(self) -> list[SeedInstance]:
        return [item for item in self.instances if item.accepted]

    @property
    def rejected(self) -> list[SeedInstance]:
        return [item for item in self.instances if not item.accepted]

    def summary(self) -> dict[str, object]:
        """One machine-readable row per photograph, for ``photographs.csv``."""
        reasons: dict[str, int] = {}
        for item in self.rejected:
            for reason in item.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        accepted = self.accepted
        sides = [item.window[2] for item in accepted]
        areas = [item.area for item in accepted]
        return {
            "path": self.path,
            "seed_type": self.seed_type,
            "sub_variety": self.sub_variety,
            "stem": self.stem,
            "width": self.width,
            "height": self.height,
            "megapixels": round(self.width * self.height / 1e6, 2),
            "seed_area_median_px": round(float(self.seed_area), 1),
            "num_detections": len(self.instances),
            "num_accepted": len(accepted),
            "num_rejected": len(self.rejected),
            "num_split_from_clusters": sum(
                1 for item in accepted if item.origin == "split"
            ),
            "num_distractors_suppressed": sum(
                1 for item in accepted if item.distractor_fraction > 0
            ),
            "num_window_shifted": sum(1 for item in accepted if item.window_shifted),
            "crop_side_median": float(np.median(sides)) if sides else 0.0,
            "crop_side_min": int(min(sides)) if sides else 0,
            "crop_side_max": int(max(sides)) if sides else 0,
            "seed_area_p05": round(float(np.percentile(areas, 5)), 1) if areas else 0.0,
            "seed_area_p95": round(float(np.percentile(areas, 95)), 1) if areas else 0.0,
            "scene_accepted": self.accepted_scene,
            "scene_reasons": "|".join(self.scene_reasons),
            **{f"reject_{name}": count for name, count in sorted(reasons.items())},
            **self.scene,
        }


def _bootstrap_seed_area(bgr: np.ndarray, params: DetectionParams) -> float:
    """A first estimate of seed size, used only to size the support erosion.

    Chicken and egg: the support region needs an erosion margin in seed
    diameters, and the seed diameter is measured from a mask that needs the
    support region. This runs the detector once against an un-eroded support to
    break the cycle. Its answer is used for the margin only -- never for the
    per-instance area gates, which are recomputed against the real support.
    """
    scene = model_scene(bgr, support_margin_px=1)
    score, darkness, chroma, _ = foreground_score(bgr, scene)
    mask = binarise(score, darkness, chroma, scene, params)
    return estimate_seed_area(mask)


def segment_photograph(
    bgr: np.ndarray,
    params: DetectionParams,
    policy: CropPolicy,
    gate: SceneGate,
    path: str = "",
    seed_type: str = "",
    sub_variety: str = "",
    stem: str = "",
    keep_arrays: bool = False,
) -> PhotographResult:
    """Detect, measure and adjudicate every seed in one photograph.

    Adjudication runs in **two passes**, and the split is not cosmetic. Pass one
    applies the size and cluster tests, which decide whether the object is a seed
    at all and need nothing but the frame's own median seed area. Pass two scores
    shape, focus and contrast against the median of the seed-sized population --
    which does not exist until pass one has said which detections are seed-sized.

    That matters because every one of those four quantities carries the variety
    and the magnification in its absolute value and not in its ratio: solidity is
    0.97 for a rice grain and 0.85 for a pearl-millet seed with a hilum notch, the
    variance of a Laplacian scales with contrast and with seed size, and grey
    contrast against paper is the seed's own colour. A constant threshold on any
    of them is a threshold on the variety, and a global 0.88 solidity floor
    discarded 29 real pearl-millet seeds from one photograph before this was
    split in two.

    Args:
        bgr: The photograph, full resolution, BGR.
        params: Detection thresholds.
        policy: Crop geometry.
        gate: Scene-level admission rules.
        path/seed_type/sub_variety/stem: Provenance, copied into the result.
        keep_arrays: Retain the label image and mask on the result. The
            visualiser needs them; the batch runner does not and would otherwise
            hold an 8 MP int32 array per photograph.
    """
    height, width = bgr.shape[:2]

    bootstrap = _bootstrap_seed_area(bgr, params)
    diameter = 2.0 * np.sqrt(max(bootstrap, 1.0) / np.pi)
    margin_px = int(round(max(params.support_margin_seeds * diameter, 8.0)))

    scene = model_scene(bgr, support_margin_px=margin_px)
    score, darkness, chroma, calibration = foreground_score(bgr, scene)
    mask = binarise(score, darkness, chroma, scene, params)
    seed_area = estimate_seed_area(mask)

    scene_row: dict[str, object] = {
        **scene.describe(),
        **{key: round(float(value), 5) for key, value in calibration.items()},
        "bootstrap_seed_area": round(float(bootstrap), 1),
    }

    result = PhotographResult(
        path=path,
        seed_type=seed_type,
        sub_variety=sub_variety,
        stem=stem,
        width=width,
        height=height,
        seed_area=seed_area,
        instances=[],
        scene=scene_row,
    )
    if seed_area <= 0:
        result.accepted_scene = False
        result.scene_reasons.append("no_seed_population")
        return result

    labels, records = component_instances(mask, seed_area, params)
    boxes = ndi.find_objects(labels)
    # Hoisted out of the per-instance loop: this is a median over ~8 million
    # pixels, and computing it per detection turned a 0.5 s photograph into a
    # 90 s one.
    paper_grey = float(np.median(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[scene.support]))

    instances: list[SeedInstance] = []
    for index, (box, record) in enumerate(zip(boxes, records)):
        if box is None:
            continue
        label = index + 1
        piece = labels[box] == label
        x0, y0 = box[1].start, box[0].start
        tight = (int(x0), int(y0), int(box[1].stop - x0), int(box[0].stop - y0))
        stats = measure(
            piece,
            tight,
            bgr[box],
            scene.paper_lab,
            paper_grey,
        )
        centre = ndi.center_of_mass(piece)
        instance = SeedInstance(
            index=len(instances),
            label=label,
            bbox=tight,
            centroid=(float(centre[1] + x0), float(centre[0] + y0)),
            area=float(stats["area"]),
            area_ratio=float(stats["area"]) / seed_area,
            origin=str(record["origin"]),
            cluster_size=int(record["cluster_size"]),
            solidity=float(stats["solidity"]),
            extent=float(stats["extent"]),
            eccentricity=float(stats["eccentricity"]),
            equivalent_diameter=float(stats["equivalent_diameter"]),
            major_axis=float(stats["major_axis"]),
            minor_axis=float(stats["minor_axis"]),
            orientation_deg=float(stats["orientation_deg"]),
            perimeter=float(stats["perimeter"]),
            circularity=float(stats["circularity"]),
            mean_rgb=stats["mean_rgb"],          # type: ignore[arg-type]
            std_rgb=stats["std_rgb"],            # type: ignore[arg-type]
            mean_lab=stats["mean_lab"],          # type: ignore[arg-type]
            contrast_vs_paper=float(stats["contrast_vs_paper"]),
            focus=float(stats["focus"]),
            contour=stats["contour"],            # type: ignore[arg-type]
        )

        # Size only. Shape and photometry are scored in a second pass, against
        # the population of THIS photograph -- which does not exist until the
        # size gate has said which detections are seed-sized at all.
        if instance.area_ratio < params.min_area_ratio:
            instance.reject("too_small")
        elif instance.area_ratio > params.max_area_ratio:
            instance.reject("too_large")
        elif instance.origin == "cluster":
            instance.reject("unresolved_cluster")

        x, y, box_w, box_h = tight
        instance.touches_frame = x <= 0 or y <= 0 or x + box_w >= width or y + box_h >= height
        if instance.touches_frame:
            instance.reject("out_of_frame")
        # A seed straddling the eroded support boundary is a seed whose mask was
        # cut by the erosion, not by its own edge -- its shape is wrong.
        if instance.accepted and not scene.support[
            max(y - 1, 0) : y + box_h + 1, max(x - 1, 0) : x + box_w + 1
        ].all():
            instance.reject("touches_support_edge")

        window, fits, shifted = square_window(tight, policy, (height, width))
        instance.window = window
        instance.window_shifted = shifted
        if not fits:
            instance.reject("oversize_window")

        instances.append(instance)

    # ---------------------------------------------------------------- pass two
    #
    # Shape, focus and contrast are all scored against the median of this
    # photograph's own seed-sized population. Every one of these quantities
    # carries the variety and the magnification in its absolute value and not in
    # its ratio: solidity is 0.97 for a rice grain and 0.85 for a pearl-millet
    # seed with a hilum notch, the variance of a Laplacian scales with contrast
    # and with seed size, and grey contrast against paper is the seed's own
    # colour. A constant threshold on any of them is a threshold on the variety.
    #
    # The absolute floors in `params` still apply underneath, for a shape no seed
    # could have at all.
    survivors = [item for item in instances if item.accepted]
    if survivors:
        def median_of(attribute: str) -> float:
            return float(np.median([getattr(item, attribute) for item in survivors]))

        solidity_median = median_of("solidity")
        circularity_median = median_of("circularity")
        focus_median = median_of("focus")
        contrast_median = median_of("contrast_vs_paper")
        result.scene.update(
            {
                "population_solidity_median": round(solidity_median, 4),
                "population_circularity_median": round(circularity_median, 4),
                "population_focus_median": round(focus_median, 2),
                "population_contrast_median": round(contrast_median, 2),
            }
        )
        for item in survivors:
            if item.solidity < max(
                params.min_solidity, params.solidity_relative * solidity_median
            ):
                item.reject("irregular")
            if item.circularity < max(
                params.min_circularity, params.circularity_relative * circularity_median
            ):
                item.reject("implausible_shape")
            if focus_median > 0 and item.focus < 0.20 * focus_median:
                item.reject("blurred")
            if contrast_median > 0 and item.contrast_vs_paper < 0.25 * contrast_median:
                item.reject("low_contrast")

    result.instances = instances
    if keep_arrays:
        result.labels = labels
        result.mask = mask

    accepted = result.accepted
    if scene.support_fraction < gate.min_support_fraction:
        result.accepted_scene = False
        result.scene_reasons.append("support_fraction_below_threshold")
    if len(accepted) < gate.min_instances:
        result.accepted_scene = False
        result.scene_reasons.append("too_few_seeds")
    # The reject fraction is computed over *seed-sized* candidates, not over
    # every connected component. A healthy tray produces hundreds of dust specks
    # and paper fibres below the size floor -- 185 of 385 components on
    # `AMT-1/IMG_0653` -- and counting those as failures would condemn every
    # clean photograph in the corpus.
    sized = [
        item
        for item in instances
        if params.min_area_ratio <= item.area_ratio <= params.max_area_ratio
    ]
    sized_rejected = [item for item in sized if not item.accepted]
    result.scene["num_seed_sized_candidates"] = len(sized)
    result.scene["seed_sized_reject_fraction"] = (
        round(len(sized_rejected) / len(sized), 4) if sized else 0.0
    )
    if sized and len(sized_rejected) / len(sized) > gate.max_reject_fraction:
        result.accepted_scene = False
        result.scene_reasons.append("majority_of_sized_detections_rejected")
    if not result.accepted_scene:
        # Every instance carries the scene verdict explicitly. Flipping
        # `accepted` without a reason would put reason-less rows in the manifest
        # and leave the overlay unable to say why nothing was written.
        for item in instances:
            item.reject("scene_rejected")
    return result
