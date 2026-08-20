"""Pictures an auditor can act on.

Three products, each answering a different question:

``overlay_photograph``
    *Did the detector find every seed, and only seeds?* The full photograph with
    the segmentation tinted on top and every detection outlined -- accepted in
    one colour, each rejection reason in its own. Rejections are drawn, not
    hidden: a photograph where the detector missed a row of seeds and a
    photograph where it found them and threw them away look completely different
    here, and they look identical in a count.

``rejection_gallery``
    *Was each rejection right?* One tile per rejected detection, at native
    resolution, captioned with its reason and the measurement that triggered it.

``before_after_panel``
    *Is the new crop better than the old one?* The legacy crop and the refined
    crop of the same physical seed, side by side at the same display scale, so
    the aspect-ratio distortion and the padding difference are visible rather
    than tabulated.

Everything here writes PNG through OpenCV and imports nothing from the model or
trainer packages.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.segmentation.pipeline import PhotographResult

#: BGR colours per verdict. Accepted is green; every rejection reason gets its
#: own hue so a glance at an overlay says *which* filter fired where.
VERDICT_COLOURS: dict[str, tuple[int, int, int]] = {
    "accepted": (60, 220, 60),
    "too_small": (200, 0, 200),
    "too_large": (0, 140, 255),
    "unresolved_cluster": (0, 0, 255),
    "irregular": (0, 200, 255),
    "implausible_shape": (120, 60, 255),
    "touches_support_edge": (255, 200, 0),
    "out_of_frame": (255, 120, 0),
    "oversize_window": (255, 60, 160),
    "low_contrast": (160, 160, 160),
    "blurred": (255, 255, 0),
    "scene_rejected": (40, 40, 200),
}


def _verdict(instance) -> str:
    if instance.accepted:
        return "accepted"
    # A scene-level rejection is appended last but is the reason that decided
    # the outcome, so it wins the label; otherwise the first per-instance reason
    # does.
    if "scene_rejected" in instance.reasons:
        return "scene_rejected"
    return instance.reasons[0] if instance.reasons else "scene_rejected"


def overlay_photograph(
    bgr: np.ndarray,
    result: PhotographResult,
    long_side: int = 2000,
    draw_windows: bool = True,
    tint: float = 0.35,
) -> np.ndarray:
    """The photograph with its segmentation, boxes and legend drawn on it.

    Drawn at full resolution and reduced once at the end, so a 2 px outline
    around a 30 px amaranthus seed survives the reduction instead of being
    averaged into the paper.

    Args:
        bgr: The photograph.
        result: Must carry ``mask``/``labels`` (segment with ``keep_arrays=True``).
        long_side: Output long side in pixels; ``0`` keeps full resolution.
        draw_windows: Also outline the square crop window of accepted
            detections, which is what makes the margin and any window shift
            visible.
        tint: Opacity of the mask tint.
    """
    canvas = bgr.copy()
    if result.mask is not None:
        overlay = canvas.copy()
        overlay[result.mask] = (0, 0, 255)
        canvas = cv2.addWeighted(overlay, tint, canvas, 1.0 - tint, 0.0)

    for instance in result.instances:
        verdict = _verdict(instance)
        colour = VERDICT_COLOURS.get(verdict, (255, 255, 255))
        x, y, width, height = instance.bbox
        cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 3)
        if draw_windows and instance.accepted:
            wx, wy, side, _ = instance.window
            cv2.rectangle(canvas, (wx, wy), (wx + side, wy + side), (255, 255, 255), 1)

    counts: dict[str, int] = {}
    for instance in result.instances:
        counts[_verdict(instance)] = counts.get(_verdict(instance), 0) + 1

    # The legend goes in a band ABOVE the photograph, not on top of it. An
    # overlaid panel hides whatever is under it, and "whatever is under it" is
    # exactly the evidence the figure exists to show.
    scale = max(canvas.shape[1] / 1600.0, 1.0)
    line_h = int(34 * scale)
    header = np.full((line_h * 3, canvas.shape[1], 3), 24, np.uint8)
    cv2.putText(
        header,
        f"{result.seed_type}/{result.sub_variety}/{result.stem}  {result.width}x{result.height}"
        f"  median seed {result.seed_area:.0f} px  scene "
        + ("ACCEPTED" if result.accepted_scene else "REJECTED: " + ",".join(result.scene_reasons)),
        (int(12 * scale), line_h - int(10 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72 * scale,
        (200, 255, 200) if result.accepted_scene else (120, 120, 255),
        max(int(2 * scale), 1), cv2.LINE_AA,
    )
    x = int(12 * scale)
    row = 1
    for name, count in sorted(counts.items(), key=lambda kv: (kv[0] != "accepted", -kv[1])):
        text = f"{name}: {count}"
        width = int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.66 * scale, 2)[0][0] + 34 * scale)
        if x + width > canvas.shape[1] - int(12 * scale):
            row += 1
            x = int(12 * scale)
            if row > 2:
                break
        cv2.rectangle(
            header, (x, row * line_h - int(21 * scale)), (x + int(18 * scale), row * line_h - int(5 * scale)),
            VERDICT_COLOURS.get(name, (255, 255, 255)), -1,
        )
        cv2.putText(
            header, text, (x + int(24 * scale), row * line_h - int(8 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.66 * scale, (235, 235, 235), max(int(2 * scale), 1), cv2.LINE_AA,
        )
        x += width
    canvas = np.vstack([header, canvas])

    if long_side and canvas.shape[1] > long_side:
        height = int(round(long_side * canvas.shape[0] / canvas.shape[1]))
        canvas = cv2.resize(canvas, (long_side, height), interpolation=cv2.INTER_AREA)
    return canvas


def _tile(images: list[tuple[str, np.ndarray]], cell: int, cols: int) -> np.ndarray:
    rows = max((len(images) + cols - 1) // cols, 1)
    canvas = np.full(((cell + 20) * rows, cell * cols, 3), 22, np.uint8)
    for index, (caption, image) in enumerate(images):
        if image.size == 0:
            continue
        scale = min(cell / image.shape[1], cell / image.shape[0], 8.0)
        resized = cv2.resize(
            image,
            (max(int(image.shape[1] * scale), 1), max(int(image.shape[0] * scale), 1)),
            # NEAREST: these tiles exist to be inspected pixel by pixel, and a
            # smooth upsample would invent boundary detail that is the very
            # thing being judged.
            interpolation=cv2.INTER_NEAREST,
        )
        x = (index % cols) * cell
        y = (index // cols) * (cell + 20)
        canvas[y + 20 : y + 20 + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            canvas, caption[:34], (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (210, 225, 255), 1, cv2.LINE_AA,
        )
    return canvas


def rejection_gallery(
    bgr: np.ndarray,
    result: PhotographResult,
    per_reason: int = 8,
    cell: int = 150,
    cols: int = 10,
) -> np.ndarray:
    """Native-resolution tiles of what was thrown away, and why.

    **Stratified by reason**, not sampled from the flat list. A photograph
    typically rejects several hundred dust specks and a handful of genuine
    borderline cases, so a uniform sample is several hundred pictures of paper
    and none of the cases an auditor needs to see. Every reason that fired
    contributes up to ``per_reason`` tiles, evenly spaced within that reason.
    """
    rejected = result.rejected
    if not rejected:
        return np.full((cell, cell, 3), 22, np.uint8)

    grouped: dict[str, list] = {}
    for instance in rejected:
        grouped.setdefault(_verdict(instance), []).append(instance)

    tiles: list[tuple[str, np.ndarray]] = []
    for reason in sorted(grouped):
        members = grouped[reason]
        step = max(len(members) // per_reason, 1)
        for instance in members[::step][:per_reason]:
            x, y, width, height = instance.bbox
            pad = max(int(0.3 * max(width, height)), 4)
            patch = bgr[
                max(y - pad, 0) : y + height + pad, max(x - pad, 0) : x + width + pad
            ]
            tiles.append(
                (f"{reason} r={instance.area_ratio:.2f} s={instance.solidity:.2f}", patch)
            )
    return _tile(tiles, cell, cols)


def accepted_gallery(
    bgr: np.ndarray, result: PhotographResult, limit: int = 40, cell: int = 150, cols: int = 10
) -> np.ndarray:
    """Native-resolution tiles of the crops that will be written."""
    accepted = result.accepted
    if not accepted:
        return np.full((cell, cell, 3), 22, np.uint8)
    step = max(len(accepted) // limit, 1)
    tiles = []
    for instance in accepted[::step][:limit]:
        x, y, side, _ = instance.window
        tiles.append((f"{instance.origin} {side}px", bgr[y : y + side, x : x + side]))
    return _tile(tiles, cell, cols)


def before_after_panel(
    pairs: list[tuple[str, np.ndarray, np.ndarray]], cell: int = 190, cols: int = 6
) -> np.ndarray:
    """Legacy crop above, refined crop below, at one display scale per pair.

    The shared scale is the point. Scaling each tile to fill its cell would hide
    exactly what changed -- that the legacy crop is a tight, non-square box that
    often clips the seed's own edge, and the refined one is a square window with
    a measured paper ring around the same seed and no aspect distortion.
    """
    label_h = 18
    block_h = 2 * (cell + label_h) + label_h
    rows = max((len(pairs) + cols - 1) // cols, 1)
    canvas = np.full((block_h * rows, cell * cols, 3), 22, np.uint8)

    for index, (caption, legacy, refined) in enumerate(pairs):
        column, row = index % cols, index // cols
        longest = max(legacy.shape[0], legacy.shape[1], refined.shape[0], refined.shape[1], 1)
        scale = min(cell / longest, 8.0)
        base_y = row * block_h
        base_x = column * cell
        cv2.putText(
            canvas, caption[:30], (base_x + 4, base_y + label_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 220, 140), 1, cv2.LINE_AA,
        )
        for offset, (name, image) in enumerate((("legacy", legacy), ("refined", refined))):
            top = base_y + label_h + offset * (cell + label_h)
            resized = cv2.resize(
                image,
                (max(int(image.shape[1] * scale), 1), max(int(image.shape[0] * scale), 1)),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(
                canvas, f"{name} {image.shape[1]}x{image.shape[0]}",
                (base_x + 4, top + label_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (210, 225, 255), 1, cv2.LINE_AA,
            )
            y = top + label_h
            x = base_x + (cell - resized.shape[1]) // 2
            canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def write(path: str | Path, image: np.ndarray) -> None:
    """``cv2.imwrite`` that creates the parent directory and reports failure.

    OpenCV returns ``False`` instead of raising when it cannot encode or write,
    which in a batch job means a silently missing figure.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write {destination}")
