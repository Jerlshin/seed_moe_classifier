"""Finding the seeds: foreground scoring, binarisation and instance separation.

The foreground score, and why it has two channels
-------------------------------------------------

A seed differs from paper in two independent ways, and neither one alone covers
the corpus. Measured inside the legacy bounding boxes against the paper around
them, on seven representative photographs:

===================== ============================= =============================
photograph            relative darkness (fg / bg)   CIELAB chroma (fg / bg)
===================== ============================= =============================
Amaranthus IMG_0653   0.39 / 0.99                   4.5 / 1.0
KodoMillet IMG_0492   0.34 / 0.98                   7.1 / 1.0
Poosa33 IMG_0661      0.35 / 0.99                   5.0 / 1.0
Chinnar IMG_0689      0.61 / 0.99                   8.1 / 1.0
**MilaguSamba 0179**  **0.73 / 0.98**               **28.4 / 1.4**
===================== ============================= =============================

``MilaguSamba/IMG_0179`` is the case that decides the design: its straw-coloured
grains sit at 73 % of the paper's brightness, with the paper's own 5th
percentile at 89 %, so a luminance-only detector must choose between missing
grains and eating the paper. In chroma the same grains are **20x** further from
the paper than the paper's own spread. Conversely the near-black amaranthus
seeds are only 4.5 chroma units from a neutral sheet but 61 % darker than it.

So the score is the **pointwise maximum of two robust z-scores** -- relative
darkness against the illumination field, and CIELAB (a, b) distance from the
paper's own colour -- each standardised by the median absolute deviation of the
*paper* in that same photograph. A pixel is foreground when either channel says
so, and "says so" is measured in units of that photograph's own noise.

Instance separation
-------------------

A connected component is not a seed. Two seeds that touch produce one component,
and on this corpus that is common enough to matter: 2-8 components per
photograph exceed 1.6x the photograph's median seed area.

The separator is a distance-transform watershed, but the *decision to run it* is
what keeps it honest. Every photograph here holds one sub-variety, so seed size
within a frame is tightly unimodal -- which makes the frame's own median
component area a reliable unit. A component is split only when its area implies
two or more seeds, the expected count comes from the same ratio, and the result
is **accepted only if every fragment is itself plausible**. A split that
produces a sliver is rejected and the component is recorded as an unresolved
cluster rather than quietly emitted as two seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from src.segmentation.illumination import SceneModel


@dataclass(slots=True, frozen=True)
class DetectionParams:
    """Thresholds for foreground scoring and instance separation.

    Every default here was chosen against a measurement recorded in this
    module's docstring or in ``illumination.py``; none is inherited from a
    tutorial. All of them are Hydra keys, so an arm moves one without a code
    change.

    Attributes:
        z_threshold: How many robust standard deviations of the *paper* a pixel
            must stand out by, in whichever channel is larger. The paper's own
            MAD is ~0.02 in relative darkness and ~0.6 chroma units, so 8 sigma
            is a wide margin in noise terms and still far below the 15-30 sigma a
            real seed reaches.
        darkness_floor: Absolute floor on ``1 - grey / field``. Guards the case
            where a photograph's paper is so uniform that its MAD collapses and
            8 sigma becomes a fraction of a grey level.
        chroma_floor: Absolute floor on CIELAB distance from the paper colour,
            in the 0-255 encoding OpenCV uses. Same purpose.
        min_area_ratio: Smallest component area, as a fraction of the
            photograph's median seed area, that can be a seed. Below this is
            chaff, dust or a paper fibre -- *recorded* as such, never silently
            dropped.
        split_area_ratio: Area, in median seeds, above which a component may be
            considered for splitting. Below it a component is never split, which
            is what protects elongated grains whose distance map has several
            maxima along one seed.
        max_single_ratio: Largest area, in median seeds, that an unsplittable
            component may still be accepted as one (large) seed. Above it the
            component is an unresolved cluster or a non-seed object.
        max_area_ratio: Hard ceiling. A component above this is a paper edge, a
            pen mark or a pile, and no split is attempted.
        split_solidity: A component is only *considered* for splitting when its
            contour area over convex-hull area falls below this -- when it has a
            waist. Measured over the corpus's split candidates, every component
            the watershed resolves into two seeds sits at 0.77-0.88 and every
            large single seed sits at 0.92-0.98, so this is a gap rather than a
            cut point. It is what stops an elongated *single* seed above
            ``split_area_ratio`` from being halved along the ridge its own
            distance transform carries.
        min_fragment_ratio / max_fragment_ratio: Plausible area for a single
            watershed fragment, in median seeds. One implausible fragment
            discards the whole split.
        peak_radius_fraction: Peak suppression radius, as a fraction of the
            expected seed diameter.
        peak_height_fraction: Minimum distance-map height for a peak, as a
            fraction of the component's maximum.
        support_margin_seeds: Erosion of the paper region, in seed diameters.
        min_solidity / min_circularity: **Absolute** floors on contour area over
            convex-hull area, and on ``4*pi*A / P^2``. Deliberately low: they
            exist to catch a shape no seed could have, not to define one.
        solidity_relative / circularity_relative: The gates that actually do the
            work, as a fraction of the **photograph's own median**. Shape is a
            varietal property, not a universal one -- a pearl-millet seed has a
            concave hilum notch and measures 0.73-0.88 solidity while a rice
            grain measures 0.97, and a global constant that keeps the rice throws
            away 29 real pearl-millet seeds from a single photograph. Scoring
            each detection against the population of its own frame is the same
            principle the focus gate already uses, for the same reason: the
            absolute value carries variety and magnification, the ratio does not.
    """

    z_threshold: float = 8.0
    darkness_floor: float = 0.07
    chroma_floor: float = 4.0
    min_area_ratio: float = 0.35
    split_area_ratio: float = 1.50
    max_single_ratio: float = 3.00
    max_area_ratio: float = 8.00
    split_solidity: float = 0.93
    min_fragment_ratio: float = 0.45
    max_fragment_ratio: float = 1.80
    peak_radius_fraction: float = 0.40
    peak_height_fraction: float = 0.25
    support_margin_seeds: float = 1.50
    min_solidity: float = 0.60
    min_circularity: float = 0.15
    solidity_relative: float = 0.80
    circularity_relative: float = 0.50


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-derived sigma, floored so a degenerate MAD cannot divide."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(mad * 1.4826, 1e-3)


def foreground_score(
    bgr: np.ndarray, scene: SceneModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Per-pixel evidence that a pixel is a seed rather than paper.

    Returns:
        ``(score, darkness, chroma, stats)``. ``score`` is the pointwise maximum
        of the two robust z-scores; ``darkness`` is ``1 - grey / field`` clipped
        at zero; ``chroma`` is the CIELAB ``(a, b)`` distance from the paper's
        own colour; ``stats`` records the four calibration numbers so a
        photograph's manifest row says how its threshold was set.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    darkness = np.clip(1.0 - grey / scene.field, 0.0, None)
    chroma = np.hypot(
        lab[:, :, 1].astype(np.float32) - scene.paper_lab[1],
        lab[:, :, 2].astype(np.float32) - scene.paper_lab[2],
    )

    support = scene.support
    dark_median, dark_sigma = _robust_scale(darkness[support])
    chroma_median, chroma_sigma = _robust_scale(chroma[support])

    score = np.maximum(
        (darkness - dark_median) / dark_sigma, (chroma - chroma_median) / chroma_sigma
    )
    stats = {
        "paper_darkness_median": dark_median,
        "paper_darkness_sigma": dark_sigma,
        "paper_chroma_median": chroma_median,
        "paper_chroma_sigma": chroma_sigma,
    }
    return score, darkness, chroma, stats


def binarise(
    score: np.ndarray,
    darkness: np.ndarray,
    chroma: np.ndarray,
    scene: SceneModel,
    params: DetectionParams,
) -> np.ndarray:
    """Threshold the score into a seed mask, then clean it.

    The absolute floors are ANDed with the z-score, not ORed: the z-score says
    "unlike this paper" and the floors say "and by an amount that could not be
    sensor noise". Both must hold.

    The cleanup is deliberately small -- a 3x3 opening to drop single-pixel
    speckle and a 5x5 closing to bridge the one- or two-pixel gaps a specular
    highlight opens along a rice husk -- followed by hole filling. A larger
    element would round the boundary, and the boundary is one of the things this
    corpus exists to preserve.
    """
    mask = (
        (score > params.z_threshold)
        & ((darkness > params.darkness_floor) | (chroma > params.chroma_floor))
        & scene.support
    ).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return ndi.binary_fill_holes(mask.astype(bool))


def _solidity(component: np.ndarray) -> float:
    """Contour area over convex-hull area, both as polygon areas.

    Mixing a pixel count with a polygon area puts this above 1 for a convex blob
    -- the polygon runs through pixel centres and under-counts by roughly half
    the perimeter -- which makes a threshold on it mean different things at
    different seed sizes. :func:`src.segmentation.instances.measure` computes the
    same quantity the same way, so the split precondition and the reported
    descriptor are one number.
    """
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    return area / hull if hull > 0 else 1.0


def estimate_seed_area(mask: np.ndarray) -> float:
    """The photograph's own unit of "one seed", in pixels.

    Every photograph here holds a single sub-variety, so component areas are
    bimodal: a tight cluster of seeds and a long tail of dust. Taking the median
    over components above 5 % of the 99th percentile discards the dust without
    needing an absolute size in pixels -- which matters because the same
    sub-variety is photographed at magnifications that differ by up to 4x
    (``Chithrakar/IMG_0161`` measures 13,672 px per seed against ``IMG_0164``'s
    3,491).

    Returns ``0.0`` when the mask holds nothing usable, which the caller reads
    as "this frame has no seed population".
    """
    labels, count = ndi.label(mask)
    if count == 0:
        return 0.0
    areas = np.bincount(labels.ravel())[1:].astype(np.float64)
    if areas.size == 0:
        return 0.0
    cutoff = max(30.0, 0.05 * float(np.percentile(areas, 99)))
    seeds = areas[areas >= cutoff]
    return float(np.median(seeds)) if seeds.size else 0.0


def split_touching(
    component: np.ndarray,
    expected: int,
    min_fragment_area: float,
    max_fragment_area: float,
    params: DetectionParams,
) -> np.ndarray | None:
    """Separate a merged component into seeds, or decline.

    The classic distance-transform watershed: a seed's interior is a maximum of
    the distance-to-background map, and the ridge between two maxima falls in
    the neck where two seeds meet.

    What makes it safe here is **when it is allowed to run and when its answer is
    believed**, both of which are measured rather than assumed.

    *Area ratio alone does not identify a merge.* On ``KodoMillet/IMG_0492``
    twelve components measure 1.5-1.85 seeds and every one of them is a single
    large seed; on ``LittleMillet/IMG_0504`` eight components measure 1.6-2.3
    seeds and most are genuine pairs. The two photographs are indistinguishable
    by area.

    *Peak count alone does not either.* An elongated rice grain has a ridged
    distance map with several local maxima along its axis: on
    ``Chinnar/IMG_0689`` **19 of 66** unambiguously single grains carry two or
    more peaks.

    The conjunction separates them cleanly, and the caller adds a third
    condition -- a **waist**. Restricting the peak test to components above
    ``split_area_ratio`` removes the elongated-grain false positives (Chinnar's
    largest component is 1.29 seeds); requiring two peaks removes the
    large-single false positives (0 of 12 on KodoMillet, 0 of 22 on PearlMillet,
    against 6 of 8 on LittleMillet); and requiring solidity below
    ``split_solidity`` covers the case the first two miss, an elongated single
    seed that is *also* large enough to be a candidate, whose distance ridge does
    supply two peaks. Every fragment is then checked for plausible area, and one
    implausible fragment rejects the whole split -- the caller records an
    unresolved cluster instead of emitting halves nobody verified.

    Args:
        component: Boolean mask of one connected component, in its own box.
        expected: Seeds the area ratio implies. Sizes the peak-suppression
            radius; it does not force a count.
        min_fragment_area / max_fragment_area: Plausible area for one seed, in
            pixels. A split producing anything outside this is discarded whole.
        params: Supplies the peak radius and height fractions.

    Returns:
        An ``int32`` label image over ``component``'s box with labels ``1..k``,
        or ``None`` when the component could not be resolved.
    """
    if expected < 2:
        return None

    distance = cv2.distanceTransform(
        component.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    peak_height = float(distance.max())
    if peak_height <= 0:
        return None

    # The suppression radius is derived from the component's own area and the
    # expected count, so it tracks the seed size of *this* photograph rather
    # than a pixel constant -- the corpus spans 30 px amaranthus to 150 px rice.
    per_seed = component.sum() / float(expected)
    radius = max(int(round(params.peak_radius_fraction * 2.0 * np.sqrt(per_seed / np.pi))), 3)
    coordinates = peak_local_max(
        distance,
        min_distance=radius,
        labels=component.astype(np.uint8),
        threshold_abs=params.peak_height_fraction * peak_height,
        exclude_border=False,
    )
    if len(coordinates) < 2:
        return None

    markers = np.zeros(component.shape, dtype=np.int32)
    for index, (row, column) in enumerate(coordinates, start=1):
        markers[row, column] = index
    # `-distance` makes the seed interiors the basins; `mask` keeps the flood
    # inside the component so no background pixel is claimed. skimage's
    # watershed labels every masked pixel, unlike OpenCV's, which writes a -1
    # boundary that would have to be reassigned afterwards.
    labels = watershed(-distance, markers, mask=component)

    result = np.zeros(component.shape, dtype=np.int32)
    for index in range(1, len(coordinates) + 1):
        region = labels == index
        area = float(region.sum())
        if area < min_fragment_area or area > max_fragment_area:
            return None
        result[region] = index
    if result.max() < 2:
        return None
    return result


def component_instances(
    mask: np.ndarray, seed_area: float, params: DetectionParams
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Label every candidate seed, splitting merged components where warranted.

    The decision tree, in the order it runs:

    1. ``area <= split_area_ratio`` seeds -- one instance, ``origin="component"``.
       Nothing at or below this size is ever split.
    2. Above it, at or below ``max_area_ratio``, **and concave** (solidity below
       ``split_solidity``) -- attempt :func:`split_touching`. On success each
       fragment becomes its own instance with ``origin="split"``. A convex
       component is never split however large it is: two touching seeds leave a
       waist and one large seed does not.
    3. A failed split at or below ``max_single_ratio`` -- one instance with
       ``origin="oversized_single"``. This is the ``PearlMillet/IMG_0510`` case:
       22 components between 1.5 and 2.6 median seeds, none with two distance
       peaks, all of them single seeds carrying an attached husk. Rejecting them
       for being large would throw away 7 % of that photograph on a measurement
       that says the opposite.
    4. A failed split above ``max_single_ratio``, or anything above
       ``max_area_ratio`` -- ``origin="cluster"``, which the caller rejects with a
       reason and records.

    Returns:
        ``(labels, records)``. ``labels`` is an ``int32`` image whose non-zero
        values index ``records`` one-to-one (label ``i`` is ``records[i - 1]``).
        Each record carries ``origin`` and ``cluster_size``, both of which reach
        the manifest -- so "how many of these crops came out of a merged blob"
        is answerable from the file rather than from the log.
    """
    labels, count = ndi.label(mask)
    output = np.zeros(mask.shape, dtype=np.int32)
    records: list[dict[str, object]] = []
    if count == 0 or seed_area <= 0:
        return output, records

    boxes = ndi.find_objects(labels)
    next_label = 1

    def emit(box, region: np.ndarray, origin: str, cluster: int, area: float, ratio: float) -> None:
        nonlocal next_label
        output[box][region] = next_label
        records.append(
            {
                "origin": origin,
                "cluster_size": cluster,
                "parent_area": area,
                "parent_ratio": ratio,
            }
        )
        next_label += 1

    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue
        piece = labels[box] == index
        area = float(piece.sum())
        ratio = area / seed_area

        if ratio <= params.split_area_ratio:
            emit(box, piece, "component", 1, area, ratio)
            continue

        concave = _solidity(piece) < params.split_solidity
        split = (
            split_touching(
                piece,
                int(round(ratio)),
                params.min_fragment_ratio * seed_area,
                params.max_fragment_ratio * seed_area,
                params,
            )
            if concave and ratio <= params.max_area_ratio
            else None
        )
        if split is not None:
            fragments = int(split.max())
            for fragment in range(1, fragments + 1):
                region = split == fragment
                if not region.any():
                    continue
                emit(box, region, "split", fragments, area, ratio)
            continue

        origin = "oversized_single" if ratio <= params.max_single_ratio else "cluster"
        emit(box, piece, origin, int(round(ratio)) if origin == "cluster" else 1, area, ratio)

    return output, records
