"""The photometric model of one photograph: what is paper, and how bright is it.

Why this file exists at all
---------------------------

The obvious way to find dark seeds on white paper is a global threshold, and on
this corpus it does not work. Measured over all 99 photographs in
``RAW_Samples`` with OpenCV's own Otsu at quarter resolution, the "foreground"
fraction it returns spans **0.9 % to 64.7 %**: on frames whose seeds cover only a
few percent of the paper, Otsu's between-class variance is maximised by
splitting the *illumination gradient of the paper itself*, not the seeds. Three
photographs (``LittleMillet/IMG_0505``, ``FoxtailMillet/IMG_0514``, ``0515``)
return above 54 % foreground while being ordinary, well-exposed trays.

Two measured properties of the corpus drive the model here:

* **The paper is not evenly lit.** Median background is 187/255 but the centre
  sits a median 4.3 % brighter than the border (range -13.3 % to +9.6 %), so a
  single number cannot describe "the paper" across one frame. The fix is a
  smooth illumination field and a *relative* deficit against it.

* **Not every photograph is a tray of seeds.** Three of the 99 are a seed
  packet on a wooden table (``Poosa33/IMG_0668``), a near-empty sheet
  (``Poosa33/IMG_0667``) and a labelled ziplock bag (``Kullakar/IMG_0713``).
  They are not defects to be silently skipped -- they are scenes whose support
  region is small and whose seed population is implausible, and
  :func:`support_region` is what makes that measurable rather than a judgement
  call.

Both quantities are computed at a **downscaled** resolution and resampled up.
Illumination and paper extent are low-frequency by construction, so nothing is
lost, and a 16x downscale turns a 200 px-kernel morphological closing on an
8 MP frame from seconds into milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

#: Long side, in pixels, of the reduced image the low-frequency model is built
#: on. The illumination field and the paper extent both vary on the scale of the
#: whole frame; the seeds -- 30-150 px -- are what must *not* survive into the
#: field, and a 16x reduction is what removes them.
DOWNSCALE = 16

#: Structuring-element side, in reduced pixels, of the closing that removes the
#: seeds from the illumination field. At ``DOWNSCALE = 16`` a 31 px element
#: spans 496 native pixels, comfortably above the 149 px widest seed bounding
#: box measured on this corpus, so no seed can leave a dent in the field.
CLOSE_KSIZE = 31


@dataclass(slots=True)
class SceneModel:
    """Everything about a photograph that is not a seed.

    Attributes:
        field: Per-pixel estimate of what the paper's grey level *would* be with
            no seed on it, same shape as the image. Never zero.
        support: Boolean mask of the paper the seeds sit on, already eroded by
            :attr:`support_margin_px` so a paper edge or its cast shadow cannot
            become a detection.
        support_fraction: Share of the frame the paper covers *before* erosion.
            The scene gate reads this: a tray photograph measures 0.97-1.00 on
            this corpus, and the three non-tray frames measure 0.58, 0.63, 0.65.
        support_margin_px: How far the support was eroded, in native pixels.
        paper_lab: Median CIELAB ``(L, a, b)`` of the paper, in OpenCV's 0-255
            encoding. The chroma channel of the foreground score is measured
            against ``(a, b)``.
        paper_grey_p50 / paper_grey_p05: Robust brightness percentiles of the
            paper, reported so a run's own artifacts say how the scene was lit.
    """

    field: np.ndarray
    support: np.ndarray
    support_fraction: float
    support_margin_px: int
    paper_lab: tuple[float, float, float]
    paper_grey_p50: float
    paper_grey_p05: float
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, float | int | list[str]]:
        """JSON-safe summary, written into the per-photograph manifest row."""
        return {
            "support_fraction": round(float(self.support_fraction), 4),
            "support_margin_px": int(self.support_margin_px),
            "paper_L": round(float(self.paper_lab[0]), 2),
            "paper_a": round(float(self.paper_lab[1]), 2),
            "paper_b": round(float(self.paper_lab[2]), 2),
            "paper_grey_p50": round(float(self.paper_grey_p50), 2),
            "paper_grey_p05": round(float(self.paper_grey_p05), 2),
            "scene_notes": list(self.notes),
        }


def resolve_downscale(
    shape: tuple[int, ...], downscale: int = DOWNSCALE, ksize: int = CLOSE_KSIZE
) -> int:
    """The reduction factor to actually use for this image.

    :data:`DOWNSCALE` is sized for the corpus's 8 MP photographs, where a 16x
    reduction leaves a ~210 x 150 image and a 31 px element covers 15 % of its
    short side. On a small image the same pair inverts: the element spans more
    than half the reduced frame, the closing flattens the illumination gradient
    it was supposed to measure, and the support mask's boundary is quantised so
    coarsely that a 26 px paper edge disappears into a single reduced pixel.

    The rule keeps the reduced short side at least ``4 * ksize``, so the element
    is always a small neighbourhood rather than most of the frame, and never
    reduces *more* than asked. It is a floor on resolution, not a policy: on
    every photograph in the corpus it returns ``downscale`` unchanged.
    """
    short = min(int(value) for value in shape[:2])
    return max(1, min(int(downscale), short // max(4 * int(ksize), 1) or 1))


def _reduce(image: np.ndarray, downscale: int = DOWNSCALE) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(width // downscale, 8), max(height // downscale, 8)),
        interpolation=cv2.INTER_AREA,
    )


def illumination_field(
    grey: np.ndarray, downscale: int = DOWNSCALE, ksize: int = CLOSE_KSIZE
) -> np.ndarray:
    """What the paper's grey level would be with no seed on it.

    A morphological **closing** -- dilate then erode -- removes structures darker
    than their surroundings and smaller than the structuring element, which is
    exactly what a seed is. The median filter afterwards removes the residual
    ridges a closing leaves where two seeds are close together.

    Returned as ``float32`` and floored at 1.0 so the caller can divide by it
    without guarding.

    Args:
        grey: Single-channel ``uint8`` image.
        downscale: Reduction factor the closing runs at. The field is
            low-frequency, so this is a speed decision, not an accuracy one.
        ksize: Closing element side, in *reduced* pixels. It must exceed the
            widest seed; see :data:`CLOSE_KSIZE`.
    """
    height, width = grey.shape[:2]
    downscale = resolve_downscale(grey.shape, downscale, ksize)
    small = _reduce(grey, downscale)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, element)
    closed = cv2.medianBlur(closed, 9)
    full = cv2.resize(closed, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.maximum(full.astype(np.float32), 1.0)


def support_region(
    bgr: np.ndarray,
    margin_px: int,
    downscale: int = DOWNSCALE,
    chroma_limit: float = 18.0,
) -> tuple[np.ndarray, float]:
    """The sheet of paper the seeds sit on, eroded by ``margin_px``.

    Built as the largest connected region that is both **bright** and
    **neutral**, with its holes filled -- the holes are the seeds, and filling
    them is what stops a seed from being carved out of its own support.

    The erosion is the load-bearing part. Without it the paper's own edge, and
    the shadow it casts on the table, survive as a single enormous component:
    on ``PearlMillet/IMG_0510`` that edge is 34x the median seed area and on
    ``Poosa33/IMG_0665`` it is 102x. Eroding by roughly one seed diameter
    removes the boundary band without touching any seed that is actually on the
    paper.

    Args:
        bgr: Full-resolution BGR image.
        margin_px: Erosion radius in *native* pixels. Callers pass a multiple of
            the estimated seed size; see
            :func:`~src.segmentation.detect.estimate_seed_area`.
        downscale: Reduction factor the region is computed at.
        chroma_limit: Maximum CIELAB chroma for a pixel to count as paper. The
            wooden table in the three non-tray frames is far above this.

    Returns:
        ``(support, fraction)`` -- the eroded boolean mask at full resolution,
        and the *un-eroded* share of the frame the paper covered.
    """
    height, width = bgr.shape[:2]
    downscale = resolve_downscale(bgr.shape, downscale)
    small = _reduce(bgr, downscale)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    chroma = np.hypot(
        lab[:, :, 1].astype(np.float32) - 128.0, lab[:, :, 2].astype(np.float32) - 128.0
    )
    # The 60th percentile, not the mean: on a frame that is mostly paper the
    # percentile *is* the paper, and on one that is mostly table it is still
    # above the table. The 0.75 factor admits the darker (shaded) end of the
    # sheet, and the absolute floor of 90 stops a frame with no paper at all
    # from promoting a dark table to "bright".
    bright = lightness > max(float(np.percentile(lightness, 60)) * 0.75, 90.0)
    neutral = chroma < float(chroma_limit)
    candidate = (bright & neutral).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return np.ones((height, width), dtype=bool), 0.0

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = ndi.binary_fill_holes(labels == largest)
    fraction = float(region.mean())

    reduced_margin = max(int(round(margin_px / downscale)), 1)
    eroded = cv2.erode(
        region.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reduced_margin + 1,) * 2),
    )
    full = cv2.resize(eroded, (width, height), interpolation=cv2.INTER_NEAREST)
    return full.astype(bool), fraction


def model_scene(bgr: np.ndarray, support_margin_px: int) -> SceneModel:
    """Build the illumination field, the support region and the paper colour.

    ``support_margin_px`` is the only free parameter and it has a physical
    meaning: the width of the band along the paper's edge that must not produce
    detections. Callers derive it from the seed size measured on this very
    photograph, so a frame of 30 px amaranthus and a frame of 150 px rice get
    proportionate margins.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    field = illumination_field(grey)
    support, fraction = support_region(bgr, support_margin_px)

    notes: list[str] = []
    if not support.any():
        # Eroding a thin support can empty it. Fall back to the whole frame and
        # say so rather than returning a model that detects nothing.
        notes.append("support_empty_after_erosion")
        support = np.ones_like(support)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    paper = support
    paper_lab = (
        float(np.median(lab[:, :, 0][paper])),
        float(np.median(lab[:, :, 1][paper])),
        float(np.median(lab[:, :, 2][paper])),
    )
    return SceneModel(
        field=field,
        support=support,
        support_fraction=fraction,
        support_margin_px=int(support_margin_px),
        paper_lab=paper_lab,
        paper_grey_p50=float(np.median(grey[paper])),
        paper_grey_p05=float(np.percentile(grey[paper], 5)),
        notes=notes,
    )
