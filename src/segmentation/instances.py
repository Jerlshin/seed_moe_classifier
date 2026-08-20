"""One detection: what it is, whether to keep it, and how to cut it out.

The crop policy, and the two things it refuses to do
---------------------------------------------------

**It does not squash.** The corpus this replaced is tight, non-square bounding
boxes -- only 3.4 % of its 9,357 crops are square -- which the stage-2 transform
resizes with an explicit ``(H, W)`` pair, i.e. it stretches each crop to a
square by a factor that depends on the seed's *orientation in the photograph*.
A rice grain lying at 0 degrees has a 3:1 box and is compressed 3x along its
length; the same grain at 45 degrees has a square box and is not compressed at
all. That turns a rigid rotation of the object into a shape change, which is
precisely the cue a fine-grained variety task needs and precisely what the
dihedral augmentation assumes is *not* happening. A square crop taken from the
photograph has no such factor.

**It does not mask.** The seed is emitted with the paper around it. Zeroing the
background would delete the boundary contrast that says where the seed ends,
which on a 40 px amaranthus seed is a substantial share of the information in
the file. What *is* removed is other seeds -- see :func:`suppress_distractors`.

The margin is measured, not chosen. Matching recovered legacy boxes against
detected tight boxes over 1,107 pairs, the legacy crops carry a median 5-6 px of
padding, i.e. ~14 % of the tight side on the small classes. A square window at
``margin = 0.12`` reproduces that ring while guaranteeing the whole boundary is
inside the frame. The cost of the margin is measured too: neighbour intrusion
into the window rises from 1.6 % at margin 0 to 2.9 % at 0.12 and 6.5 % at 0.30,
and out-of-frame windows stay under 0.8 % throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

#: Every reason a detection can fail to become a crop. Exposed as a constant
#: because the audit report tabulates all of them and a report that silently
#: omitted a category would be the failure this package exists to prevent.
#:
#: Nothing is dropped without one of these: a rejected detection still gets a
#: manifest row, a bounding box in the overlay and a tile in the rejection
#: gallery.
REJECTION_REASONS = (
    "too_small",            # below min_area_ratio of THIS frame's median seed
    "too_large",            # above max_area_ratio: a blob, a mark, a paper edge
    "unresolved_cluster",   # merged seeds the watershed declined to split
    "irregular",            # solidity far below THIS photograph's own median
    "implausible_shape",    # a sliver: too little area for its perimeter to be a seed
    "touches_support_edge", # part of the seed lies outside the eroded paper
    "out_of_frame",         # the seed's own box is cut by the photograph's border
    "oversize_window",      # the square window is larger than the frame's short side
    "low_contrast",         # barely distinguishable from paper, against its peers
    "blurred",              # focus measure far below THIS photograph's own median
    "scene_rejected",       # the photograph is not a tray of seeds; see SceneGate
)


@dataclass(slots=True)
class CropPolicy:
    """How a detection becomes a square image file.

    Attributes:
        margin: Padding on each side, as a fraction of the tight box's longer
            side. The square window is ``max(w, h) * (1 + 2 * margin)``.
        min_side: Floor on the emitted side, in pixels. A detection whose window
            is smaller is *enlarged* around the same centre rather than
            upsampled, so the extra pixels are real paper rather than
            interpolation.
        interpolation_free: Assert that no crop is ever resampled. Kept as a
            flag so a caller that deliberately wants a fixed output size has to
            say so; the extraction pipeline never sets it False.
        suppress_distractors: Replace other seeds that intrude into the window
            with paper. See :func:`suppress_distractors`.
        distractor_dilation: How far, in pixels, the intruding seed's mask is
            grown before it is filled, so its shadow and its anti-aliased rim go
            with it.
    """

    margin: float = 0.12
    min_side: int = 32
    interpolation_free: bool = True
    suppress_distractors: bool = True
    distractor_dilation: int = 3


@dataclass(slots=True)
class SeedInstance:
    """One detected seed, its descriptors and its verdict.

    The descriptors are the ones an audit needs to decide whether a corpus is
    trustworthy without opening every file: geometry (is it seed-shaped?),
    photometry (is it a seed or a smudge?), focus (is it in the plane the camera
    was focused on?) and provenance (which photograph, which detection, was it
    a whole component or half of a merged pair?).
    """

    index: int
    label: int
    bbox: tuple[int, int, int, int]        # tight (x, y, w, h) in the photograph
    centroid: tuple[float, float]
    area: float
    area_ratio: float
    origin: str
    cluster_size: int

    # geometry
    solidity: float
    extent: float
    eccentricity: float
    equivalent_diameter: float
    major_axis: float
    minor_axis: float
    orientation_deg: float
    perimeter: float
    circularity: float

    # photometry, measured on the seed pixels only
    mean_rgb: tuple[float, float, float]
    std_rgb: tuple[float, float, float]
    mean_lab: tuple[float, float, float]
    contrast_vs_paper: float
    focus: float

    # crop
    window: tuple[int, int, int, int] = (0, 0, 0, 0)   # (x, y, side, side)
    distractor_fraction: float = 0.0
    distractor_labels: tuple[int, ...] = ()
    window_shifted: bool = False
    touches_frame: bool = False

    # verdict
    accepted: bool = True
    reasons: list[str] = field(default_factory=list)
    filename: str = ""
    contour: list[tuple[int, int]] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        """Mark this detection rejected. Reasons accumulate; none is discarded."""
        if reason not in REJECTION_REASONS:
            raise ValueError(f"Unknown rejection reason {reason!r}")
        self.accepted = False
        if reason not in self.reasons:
            self.reasons.append(reason)

    def row(self) -> dict[str, object]:
        """Flat, JSON- and CSV-safe record of this detection."""
        data = asdict(self)
        for key in ("mean_rgb", "std_rgb", "mean_lab"):
            red, green, blue = data.pop(key)
            suffix = ("r", "g", "b") if key != "mean_lab" else ("L", "a", "b")
            data[f"{key}_{suffix[0]}"] = round(float(red), 3)
            data[f"{key}_{suffix[1]}"] = round(float(green), 3)
            data[f"{key}_{suffix[2]}"] = round(float(blue), 3)
        data["bbox_x"], data["bbox_y"], data["bbox_w"], data["bbox_h"] = data.pop("bbox")
        data["window_x"], data["window_y"], data["window_side"], _ = data.pop("window")
        data["centroid_x"], data["centroid_y"] = (
            round(float(value), 2) for value in data.pop("centroid")
        )
        data["reasons"] = "|".join(data["reasons"])
        data["distractor_labels"] = "|".join(str(item) for item in data["distractor_labels"])
        data.pop("contour", None)
        return data


def measure(
    instance_mask: np.ndarray,
    box: tuple[int, int, int, int],
    bgr_window: np.ndarray,
    paper_lab: tuple[float, float, float],
    paper_grey: float,
) -> dict[str, object]:
    """Shape, colour and focus descriptors for one instance.

    ``instance_mask`` and ``bgr_window`` are both in the instance's own tight
    box, so nothing here depends on where in the photograph the seed was.

    The focus measure is the variance of the Laplacian over the seed pixels --
    the standard no-reference sharpness proxy. Its absolute value is
    meaningless (it scales with contrast and with seed size), which is why the
    pipeline compares it to the *median over the same photograph* rather than to
    a constant.
    """
    mask_u8 = instance_mask.astype(np.uint8)
    area = float(mask_u8.sum())
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else np.zeros((1, 1, 2), np.int32)

    # Both areas are polygon areas. Mixing a pixel count with a polygon area
    # puts solidity above 1 for a convex blob (the polygon runs through pixel
    # centres and so under-counts by roughly half the perimeter), which makes a
    # threshold on it mean different things at different seed sizes.
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or contour_area
    perimeter = float(cv2.arcLength(contour, True))
    height, width = instance_mask.shape

    if contour.shape[0] >= 5:
        (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
    else:
        major, minor, angle = float(max(width, height)), float(min(width, height)), 0.0
    ratio = minor / major if major > 0 else 1.0
    eccentricity = float(np.sqrt(max(1.0 - ratio * ratio, 0.0)))

    rgb = cv2.cvtColor(bgr_window, cv2.COLOR_BGR2RGB).astype(np.float32)
    lab = cv2.cvtColor(bgr_window, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = instance_mask
    grey = cv2.cvtColor(bgr_window, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_32F)

    return {
        "area": area,
        "solidity": float(contour_area / hull_area) if hull_area > 0 else 0.0,
        "extent": float(area / max(width * height, 1)),
        "eccentricity": eccentricity,
        "equivalent_diameter": float(2.0 * np.sqrt(area / np.pi)),
        "major_axis": float(major),
        "minor_axis": float(minor),
        "orientation_deg": float(angle),
        "perimeter": perimeter,
        "circularity": float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0,
        "mean_rgb": tuple(float(rgb[:, :, c][pixels].mean()) for c in range(3)),
        "std_rgb": tuple(float(rgb[:, :, c][pixels].std()) for c in range(3)),
        "mean_lab": tuple(float(lab[:, :, c][pixels].mean()) for c in range(3)),
        "contrast_vs_paper": float(paper_grey - grey[pixels].mean()),
        "focus": float(laplacian[pixels].var()),
        "contour": [
            (int(point[0][0] + box[0]), int(point[0][1] + box[1]))
            for point in cv2.approxPolyDP(contour, 0.6, True)
        ],
    }


def square_window(
    box: tuple[int, int, int, int],
    policy: CropPolicy,
    frame: tuple[int, int],
) -> tuple[tuple[int, int, int, int], bool, bool]:
    """The square crop window for a tight box.

    The window is centred on the tight box and sized from its longer side, so
    the seed's true aspect ratio survives into the file. When the window would
    leave the photograph it is **translated** back inside rather than shrunk --
    shrinking would clip the seed, translating only moves the paper ring, and
    ``side >= max(w, h)`` guarantees the translated window still contains the
    whole seed. A window larger than the frame's short side cannot be placed at
    all.

    Args:
        box: ``(x, y, w, h)`` tight bounding box.
        policy: Margin and floor.
        frame: ``(height, width)`` of the photograph.

    Returns:
        ``((x, y, side, side), fits, shifted)``. ``fits`` is False when the
        window is larger than the frame, in which case the window must not be
        written. ``shifted`` says the window was translated off-centre to stay
        inside the frame -- not a defect, but recorded, because a systematically
        off-centre crop is a thing an audit should be able to see.
    """
    x, y, width, height = box
    side = int(round(max(width, height) * (1.0 + 2.0 * policy.margin)))
    side = max(side, int(policy.min_side), width, height)
    frame_h, frame_w = frame
    if side > min(frame_h, frame_w):
        return (0, 0, min(frame_h, frame_w), min(frame_h, frame_w)), False, False

    left = int(round(x + width / 2.0 - side / 2.0))
    top = int(round(y + height / 2.0 - side / 2.0))
    shifted = left < 0 or top < 0 or left + side > frame_w or top + side > frame_h
    left = max(0, min(left, frame_w - side))
    top = max(0, min(top, frame_h - side))
    return (left, top, side, side), True, shifted


def suppress_distractors(
    crop: np.ndarray,
    window_labels: np.ndarray,
    keep_label: int,
    dilation: int,
) -> tuple[np.ndarray, float, tuple[int, ...]]:
    """Replace *other* seeds inside the window with paper.

    A square window around one seed catches a neighbour 2.9 % of the time at the
    configured margin. Leaving the neighbour in would put a second, differently
    posed instance of the same class in the frame -- which at DINO's local-crop
    scales can become the whole view -- and cropping tighter to avoid it would
    clip the target's own boundary.

    The fill is Telea inpainting from the paper immediately around the
    distractor. That is the right operator here for a reason specific to this
    imagery: the surround is a smooth, textureless sheet, so inpainting
    reproduces its gradient and its grain statistics rather than pasting a flat
    patch that would itself read as an object. The target seed's own pixels are
    never touched.

    Returns:
        ``(crop, fraction, labels)`` -- the edited copy, the share of the window
        that was replaced, and which detection labels were removed. All three go
        into the manifest, so a crop that was heavily edited is identifiable.
    """
    other = (window_labels != 0) & (window_labels != keep_label)
    if not other.any():
        return crop, 0.0, ()

    removed = tuple(int(value) for value in np.unique(window_labels[other]))
    fill = other.astype(np.uint8)
    if dilation > 0:
        fill = cv2.dilate(
            fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
        )
    # Never erase the target, even if the dilation reaches it.
    fill[window_labels == keep_label] = 0
    if not fill.any():
        return crop, 0.0, ()
    repaired = cv2.inpaint(crop, fill, 3, cv2.INPAINT_TELEA)
    return repaired, float(fill.mean()), removed


def render_crop(
    bgr: np.ndarray,
    labels: np.ndarray,
    instance: SeedInstance,
    policy: CropPolicy,
) -> np.ndarray:
    """Cut ``instance``'s square crop out of the photograph.

    No resampling happens anywhere in this function: the returned array is a
    copy of real sensor pixels, optionally with other seeds inpainted out. That
    is the whole point -- every downstream resize is then a single, explicit,
    configurable step in the transform pipeline rather than a loss baked into
    the corpus.
    """
    x, y, side, _ = instance.window
    crop = bgr[y : y + side, x : x + side].copy()
    if policy.interpolation_free and crop.shape[0] != crop.shape[1]:
        raise ValueError(
            f"Window {instance.window} did not yield a square crop ({crop.shape}); "
            "a non-square window would force a resize."
        )
    if not policy.suppress_distractors:
        return crop
    window_labels = labels[y : y + side, x : x + side]
    crop, fraction, removed = suppress_distractors(
        crop, window_labels, instance.label, policy.distractor_dilation
    )
    instance.distractor_fraction = fraction
    instance.distractor_labels = removed
    return crop
