"""Validate a refined corpus against evidence, not against its own confidence.

    python -m src.segmentation.audit
    python -m src.segmentation.audit audit.match_legacy=false      # skip the reference pass
    python -m src.segmentation.audit audit.iou_threshold=0.5

The reference annotation set
----------------------------

The hard question about any detector is "what did it miss?", and normally there
is nothing to answer it with. Here there is. **Every crop under
``Cropped_Samples`` is a byte-identical sub-image of its raw photograph** --
verified, not assumed: a full-resolution ``TM_SQDIFF`` template match returns
exactly zero and the recovered patch compares equal to the crop, for every crop
tested. So the legacy corpus is a set of 9,357 human-curated bounding boxes
whose exact coordinates are recoverable.

That turns three opinions into measurements:

* **Recall.** How many of the 9,357 curated seeds does the new detector find?
* **Novelty.** How many seeds does it find that the curated set does not
  contain -- and are they real?
* **Duplication.** Do two crops in either corpus describe the same seed?

The match is on the tight boxes at a deliberately loose IoU. The two corpora use
different padding rules -- the legacy crops carry a measured 5-6 px ring, the
refined ones a 12 % square margin -- so a strict IoU would be measuring the
margin rather than whether the same seed was found.

Everything else here needs no reference: duplicate detection, per-class
coverage, size and shape distributions, and the before/after panels.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.visualize import _tile, before_after_panel, write  # noqa: E402

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Recovering the legacy boxes
# --------------------------------------------------------------------------

def locate_crop(
    photograph: np.ndarray,
    photograph_grey: np.ndarray,
    reduced_grey: np.ndarray,
    scale: int,
    crop: np.ndarray,
) -> tuple[int, int, int, int, bool] | None:
    """Where in ``photograph`` this crop was cut from.

    Coarse-to-fine, and **verified byte-for-byte**. The coarse pass runs a
    normalised cross-correlation at ``1/scale`` resolution to get within a few
    pixels; the fine pass runs a sum-of-squared-differences in a small window
    around it; the result is then compared to the crop element by element.

    Measured on 30 amaranthus crops (the hardest case -- 40 px crops on a
    3147x2102 sheet, where thousands of positions look alike at low
    resolution): exhaustive full-resolution search recovers 30/30 at 79 ms per
    crop, ``scale=2`` recovers 28/30 at 20 ms, ``scale=4`` only 14/30 at 6 ms.
    Hence ``scale=2`` with a full-resolution fallback for the misses, which is
    exact everywhere and costs ~25 ms per crop.

    Returns ``(x, y, w, h, exact)``, or ``None`` when the crop is larger than
    the photograph.
    """
    crop_grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = crop.shape[:2]
    if height > photograph.shape[0] or width > photograph.shape[1]:
        return None

    def verify(x: int, y: int) -> bool:
        patch = photograph[y : y + height, x : x + width]
        return patch.shape == crop.shape and bool(np.array_equal(patch, crop))

    if scale > 1:
        small = cv2.resize(
            crop_grey,
            (max(width // scale, 2), max(height // scale, 2)),
            interpolation=cv2.INTER_AREA,
        )
        if small.shape[0] <= reduced_grey.shape[0] and small.shape[1] <= reduced_grey.shape[1]:
            _, _, _, best = cv2.minMaxLoc(
                cv2.matchTemplate(reduced_grey, small, cv2.TM_CCOEFF_NORMED)
            )
            guess_x, guess_y = best[0] * scale, best[1] * scale
            pad = 4 * scale
            x0, y0 = max(guess_x - pad, 0), max(guess_y - pad, 0)
            window = photograph_grey[y0 : guess_y + height + pad, x0 : guess_x + width + pad]
            if window.shape[0] >= height and window.shape[1] >= width:
                _, _, location, _ = cv2.minMaxLoc(
                    cv2.matchTemplate(window, crop_grey, cv2.TM_SQDIFF)
                )
                x, y = x0 + location[0], y0 + location[1]
                if verify(x, y):
                    return int(x), int(y), int(width), int(height), True

    _, _, location, _ = cv2.minMaxLoc(
        cv2.matchTemplate(photograph_grey, crop_grey, cv2.TM_SQDIFF)
    )
    x, y = int(location[0]), int(location[1])
    return x, y, int(width), int(height), verify(x, y)


def recover_legacy_boxes(
    legacy_root: Path, raw_root: Path, scale: int = 2
) -> dict[str, list[dict[str, object]]]:
    """Exact bounding boxes for every legacy crop, keyed by ``sub_variety/stem``.

    Crops whose source photograph is missing are reported under
    ``__unmatched_source__`` rather than skipped, because a legacy crop with no
    raw photograph is itself a finding.
    """
    by_source: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(legacy_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stem = path.stem.rsplit("_bbox", 1)[0]
        by_source[f"{path.parent.parent.name}/{path.parent.name}/{stem}"].append(path)

    boxes: dict[str, list[dict[str, object]]] = {}
    for key, crops in sorted(by_source.items()):
        seed_type, sub_variety, stem = key.split("/")
        photograph_path = raw_root / seed_type / sub_variety / f"{stem}.JPG"
        if not photograph_path.exists():
            candidates = list((raw_root / seed_type / sub_variety).glob(f"{stem}.*"))
            if not candidates:
                boxes.setdefault("__unmatched_source__", []).append(
                    {"key": key, "num_crops": len(crops)}
                )
                continue
            photograph_path = candidates[0]

        photograph = cv2.imread(str(photograph_path), cv2.IMREAD_COLOR)
        grey = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)
        reduced = cv2.resize(
            grey,
            (grey.shape[1] // scale, grey.shape[0] // scale),
            interpolation=cv2.INTER_AREA,
        ) if scale > 1 else grey

        found: list[dict[str, object]] = []
        for crop_path in sorted(crops):
            crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            located = locate_crop(photograph, grey, reduced, scale, crop)
            if located is None:
                continue
            x, y, width, height, exact = located
            found.append(
                {
                    "file": crop_path.name,
                    "x": x, "y": y, "w": width, "h": height,
                    "exact": exact,
                }
            )
        boxes[key] = found
        LOGGER.info(
            "  %-46s %4d crops, %4d located exactly", key, len(crops),
            sum(1 for item in found if item["exact"]),
        )
    return boxes


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union between two ``(n, 4)`` xywh box sets."""
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    lx1, ly1 = left[:, 0][:, None], left[:, 1][:, None]
    lx2, ly2 = lx1 + left[:, 2][:, None], ly1 + left[:, 3][:, None]
    rx1, ry1 = right[:, 0][None, :], right[:, 1][None, :]
    rx2, ry2 = rx1 + right[:, 2][None, :], ry1 + right[:, 3][None, :]

    inter_w = np.clip(np.minimum(lx2, rx2) - np.maximum(lx1, rx1), 0, None)
    inter_h = np.clip(np.minimum(ly2, ry2) - np.maximum(ly1, ry1), 0, None)
    intersection = inter_w * inter_h
    union = (
        left[:, 2][:, None] * left[:, 3][:, None]
        + right[:, 2][None, :] * right[:, 3][None, :]
        - intersection
    )
    return intersection / np.maximum(union, 1.0)


def greedy_match(scores: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one matching, best pair first.

    Greedy rather than Hungarian on purpose: the boxes here are near-disjoint
    (seeds do not overlap), so the assignment is essentially unique and the
    optimal matching would be identical. Greedy makes the result easy to read
    and its failure mode -- an ambiguous pair -- visible as a low-IoU match
    rather than hidden inside a cost matrix.
    """
    pairs: list[tuple[int, int, float]] = []
    if scores.size == 0:
        return pairs
    work = scores.copy()
    while True:
        index = int(np.argmax(work))
        row, column = divmod(index, work.shape[1])
        value = float(work[row, column])
        if value < threshold:
            break
        pairs.append((row, column, value))
        work[row, :] = -1.0
        work[:, column] = -1.0
    return pairs


def self_overlaps(boxes: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Pairs within one box set that overlap more than ``threshold``.

    On a corpus of one-seed-per-file crops this should be empty. A non-empty
    result means two files describe the same physical seed, which inflates every
    count and puts the same object on both sides of a train/test split.
    """
    if boxes.shape[0] < 2:
        return []
    scores = iou_matrix(boxes, boxes)
    np.fill_diagonal(scores, 0.0)
    rows, columns = np.where(np.triu(scores) > threshold)
    return [(int(r), int(c), float(scores[r, c])) for r, c in zip(rows, columns)]


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def load_manifest(report_dir: Path) -> list[dict[str, object]]:
    """Read ``manifest.csv`` back with its numeric and boolean columns typed."""
    path = report_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.segmentation.extract` first."
        )
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            typed: dict[str, object] = {}
            for key, value in row.items():
                if value in {"True", "False"}:
                    typed[key] = value == "True"
                    continue
                try:
                    typed[key] = int(value)
                except (TypeError, ValueError):
                    try:
                        typed[key] = float(value)
                    except (TypeError, ValueError):
                        typed[key] = value
            rows.append(typed)
    return rows


def distribution(values: np.ndarray) -> dict[str, float]:
    """Percentile summary used for every numeric column in the report."""
    if values.size == 0:
        return {key: 0.0 for key in ("min", "p05", "p25", "median", "p75", "p95", "max", "mean")}
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def corpus_statistics(root: Path) -> dict[str, object]:
    """Size, aspect and count statistics of an image-folder corpus.

    Reads file headers only -- ``PIL.Image.open`` is lazy -- so a 13,000-image
    corpus is a second rather than a minute.
    """
    from PIL import Image

    widths: list[int] = []
    heights: list[int] = []
    per_class: dict[str, int] = defaultdict(int)
    sources: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(path) as image:
            widths.append(image.size[0])
            heights.append(image.size[1])
        per_class[f"{path.parent.parent.name}/{path.parent.name}"] += 1
        sources.add(f"{path.parent.name}/{path.stem.rsplit('_bbox', 1)[0]}")

    width = np.asarray(widths, dtype=np.float64)
    height = np.asarray(heights, dtype=np.float64)
    if width.size == 0:
        return {"root": str(root), "num_images": 0}
    return {
        "root": str(root),
        "num_images": int(width.size),
        "num_classes": len(per_class),
        "num_source_photographs": len(sources),
        "crops_per_class": dict(sorted(per_class.items())),
        "width": distribution(width),
        "height": distribution(height),
        "area": distribution(width * height),
        "aspect_ratio": distribution(width / np.maximum(height, 1)),
        "square_fraction": float((width == height).mean()),
        "both_sides_under_64_fraction": float(((width < 64) & (height < 64)).mean()),
        "min_side": distribution(np.minimum(width, height)),
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="segmentation")
def main(cfg: DictConfig) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report_dir = Path(cfg.paths.report_dir).expanduser()
    raw_root = Path(cfg.paths.raw_root).expanduser()
    legacy_root = Path(cfg.paths.legacy_root).expanduser()
    refined_root = Path(cfg.paths.output_root).expanduser()

    rows = load_manifest(report_dir)
    accepted = [row for row in rows if row["accepted"]]
    LOGGER.info("Manifest: %s detections, %s written crops.", len(rows), len(accepted))

    report: dict[str, object] = {
        "report_dir": str(report_dir),
        "refined": corpus_statistics(refined_root),
        "legacy": corpus_statistics(legacy_root) if legacy_root.exists() else {},
    }

    # ---------------------------------------------------------------- quality
    numeric = {
        "window_side": "crop side (px)",
        "area": "seed mask area (px)",
        "area_ratio": "area / photograph median seed",
        "solidity": "solidity",
        "circularity": "circularity",
        "eccentricity": "eccentricity",
        "contrast_vs_paper": "grey contrast against paper",
        "focus": "variance of Laplacian",
        "distractor_fraction": "share of crop inpainted",
    }
    report["accepted_quality"] = {
        key: distribution(np.asarray([float(row[key]) for row in accepted]))
        for key in numeric
        if accepted and key in accepted[0]
    }
    report["origin_counts"] = {
        name: sum(1 for row in accepted if row["origin"] == name)
        for name in sorted({str(row["origin"]) for row in accepted})
    }

    # --------------------------------------------------------- self-duplicates
    duplicates: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in accepted:
        grouped[f"{row['sub_variety']}/{row['source_photograph']}"].append(row)
    for key, members in sorted(grouped.items()):
        boxes = np.asarray(
            [[row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]] for row in members],
            dtype=np.float64,
        )
        for left, right, score in self_overlaps(boxes, float(cfg.audit.duplicate_iou)):
            duplicates.append(
                {
                    "photograph": key,
                    "a": members[left]["filename"],
                    "b": members[right]["filename"],
                    "iou": round(score, 3),
                }
            )
    report["duplicate_detections"] = {
        "count": len(duplicates),
        "iou_threshold": float(cfg.audit.duplicate_iou),
        "pairs": duplicates[:100],
    }

    # --------------------------------------------------------- legacy reference
    if bool(cfg.audit.match_legacy) and legacy_root.exists() and raw_root.exists():
        cache = report_dir / "legacy_boxes.json"
        if cache.exists():
            LOGGER.info("Reusing recovered legacy boxes from %s", cache)
            legacy = json.loads(cache.read_text(encoding="utf-8"))
        else:
            LOGGER.info(
                "Recovering exact legacy bounding boxes by template matching "
                "(this is the reference set; ~4 minutes)."
            )
            started = time.time()
            legacy = recover_legacy_boxes(legacy_root, raw_root, int(cfg.audit.coarse_scale))
            LOGGER.info("Recovered in %.0f s.", time.time() - started)
            cache.write_text(json.dumps(legacy), encoding="utf-8")

        threshold = float(cfg.audit.iou_threshold)
        matched = missed = novel = 0
        legacy_total = 0
        inexact = 0
        per_photograph: list[dict[str, object]] = []
        missed_boxes: list[dict[str, object]] = []
        legacy_duplicates: list[dict[str, object]] = []
        unreliable: list[dict[str, object]] = []

        for key, entries in sorted(legacy.items()):
            if key == "__unmatched_source__":
                continue
            seed_type, sub_variety, stem = key.split("/")
            legacy_boxes = np.asarray(
                [[item["x"], item["y"], item["w"], item["h"]] for item in entries],
                dtype=np.float64,
            )
            # A photograph whose crops are NOT byte-exact sub-images of it is a
            # photograph the reference cannot describe. Four of the 81 are like
            # that -- Chithrakar IMG_0161/0162 and Kullakar IMG_0711/0712 are
            # stored as portrait 3024x4032 originals while their crops were cut
            # from a rotated, differently-cropped version that is no longer in
            # RAW_Samples (mean absolute pixel difference 15-22 at the recovered
            # location, against exactly 0 everywhere else). Their recovered
            # boxes are wrong, so scoring recall against them would report a
            # detector failure that is really a corpus inconsistency. They are
            # counted and named, and excluded from the recall denominator.
            not_exact = sum(1 for item in entries if not item["exact"])
            if entries and not_exact > 0.5 * len(entries):
                inexact += not_exact
                unreliable.append(
                    {
                        "photograph": key,
                        "legacy_crops": len(entries),
                        "not_byte_exact": not_exact,
                        "refined_crops": len(grouped.get(f"{sub_variety}/{stem}", [])),
                    }
                )
                continue
            inexact += not_exact
            legacy_total += len(entries)

            members = grouped.get(f"{sub_variety}/{stem}", [])
            refined_boxes = np.asarray(
                [[row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]] for row in members],
                dtype=np.float64,
            )
            scores = iou_matrix(legacy_boxes, refined_boxes)
            pairs = greedy_match(scores, threshold)
            matched_legacy = {pair[0] for pair in pairs}
            matched_refined = {pair[1] for pair in pairs}
            matched += len(pairs)
            missed += len(entries) - len(matched_legacy)
            novel += len(members) - len(matched_refined)

            for index in range(len(entries)):
                if index not in matched_legacy:
                    missed_boxes.append(
                        {
                            "seed_type": seed_type, "sub_variety": sub_variety, "stem": stem,
                            "file": entries[index]["file"],
                            "x": entries[index]["x"], "y": entries[index]["y"],
                            "w": entries[index]["w"], "h": entries[index]["h"],
                            "best_iou": round(float(scores[index].max()), 3) if scores.size else 0.0,
                        }
                    )
            for left, right, score in self_overlaps(legacy_boxes, float(cfg.audit.duplicate_iou)):
                legacy_duplicates.append(
                    {
                        "photograph": key,
                        "a": entries[left]["file"], "b": entries[right]["file"],
                        "iou": round(score, 3),
                    }
                )
            per_photograph.append(
                {
                    "photograph": key,
                    "legacy": len(entries),
                    "refined": len(members),
                    "matched": len(pairs),
                    "legacy_only": len(entries) - len(matched_legacy),
                    "refined_only": len(members) - len(matched_refined),
                    "median_iou": round(float(np.median([p[2] for p in pairs])), 3) if pairs else 0.0,
                }
            )

        report["legacy_reference"] = {
            "iou_threshold": threshold,
            "legacy_crops_located": legacy_total,
            "legacy_crops_not_byte_exact": inexact,
            "photographs_excluded_from_reference": unreliable,
            "photographs_excluded_note": (
                "Their legacy crops are not byte-exact sub-images of the RAW_Samples "
                "photograph of the same name -- the raw file was replaced by a differently "
                "oriented or cropped version after the crops were cut -- so the recovered "
                "boxes do not describe these photographs and cannot score a detector."
            ),
            "matched": matched,
            "recall_of_legacy": round(matched / max(legacy_total, 1), 4),
            "legacy_only": missed,
            "refined_only": novel,
            "legacy_duplicate_pairs": len(legacy_duplicates),
            "per_photograph": per_photograph,
            "legacy_only_examples": missed_boxes[:200],
            "legacy_duplicate_examples": legacy_duplicates[:50],
        }
        with (report_dir / "legacy_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_photograph[0]))
            writer.writeheader()
            writer.writerows(per_photograph)

        # A gallery of what the refined detector did NOT find. This is the one
        # figure that can prove seeds were missed, so it is always written when
        # anything went unmatched.
        if missed_boxes:
            tiles = []
            for entry in missed_boxes[:60]:
                photograph = cv2.imread(
                    str(raw_root / entry["seed_type"] / entry["sub_variety"] / f"{entry['stem']}.JPG")
                )
                if photograph is None:
                    continue
                pad = max(int(0.4 * max(entry["w"], entry["h"])), 5)
                tiles.append(
                    (
                        f"{entry['sub_variety']} iou={entry['best_iou']}",
                        photograph[
                            max(entry["y"] - pad, 0) : entry["y"] + entry["h"] + pad,
                            max(entry["x"] - pad, 0) : entry["x"] + entry["w"] + pad,
                        ],
                    )
                )
            if tiles:
                write(report_dir / "figures" / "legacy_only_seeds.png", _tile(tiles, 150, 10))

        # Before/after: the same physical seed, legacy crop over refined crop.
        pairs_to_draw: list[tuple[str, np.ndarray, np.ndarray]] = []
        wanted = int(cfg.audit.before_after_samples)
        for key, entries in sorted(legacy.items()):
            if key == "__unmatched_source__" or len(pairs_to_draw) >= wanted:
                continue
            seed_type, sub_variety, stem = key.split("/")
            members = grouped.get(f"{sub_variety}/{stem}", [])
            if not members or not entries:
                continue
            legacy_boxes = np.asarray(
                [[item["x"], item["y"], item["w"], item["h"]] for item in entries], dtype=np.float64
            )
            refined_boxes = np.asarray(
                [[row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]] for row in members],
                dtype=np.float64,
            )
            match = greedy_match(iou_matrix(legacy_boxes, refined_boxes), threshold)
            if not match:
                continue
            left, right, _ = match[0]
            legacy_image = cv2.imread(
                str(legacy_root / seed_type / sub_variety / str(entries[left]["file"]))
            )
            refined_image = cv2.imread(
                str(refined_root / seed_type / sub_variety / str(members[right]["filename"]))
            )
            if legacy_image is None or refined_image is None:
                continue
            pairs_to_draw.append((sub_variety, legacy_image, refined_image))
        if pairs_to_draw:
            write(report_dir / "figures" / "before_after.png", before_after_panel(pairs_to_draw))

    # -------------------------------------------------------------- coverage
    photographs = (report_dir / "photographs.csv").read_text(encoding="utf-8").splitlines()
    report["photographs_csv_rows"] = max(len(photographs) - 1, 0)

    (report_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ print
    refined_stats = report["refined"]
    legacy_stats = report["legacy"]
    LOGGER.info("\n%s", "=" * 84)
    LOGGER.info("CORPUS COMPARISON")
    LOGGER.info(
        "%-34s %18s %18s", "", "legacy", "refined",
    )
    def line(label: str, left: object, right: object) -> None:
        LOGGER.info("%-34s %18s %18s", label, left, right)

    if legacy_stats:
        line("images", legacy_stats["num_images"], refined_stats["num_images"])
        line("source photographs",
             legacy_stats["num_source_photographs"], refined_stats["num_source_photographs"])
        line("classes", legacy_stats["num_classes"], refined_stats["num_classes"])
        line("square fraction",
             f"{legacy_stats['square_fraction']:.1%}", f"{refined_stats['square_fraction']:.1%}")
        for field in ("width", "height", "area", "min_side"):
            line(f"{field} median",
                 f"{legacy_stats[field]['median']:.0f}", f"{refined_stats[field]['median']:.0f}")
        line("aspect p05/p95",
             f"{legacy_stats['aspect_ratio']['p05']:.2f}/{legacy_stats['aspect_ratio']['p95']:.2f}",
             f"{refined_stats['aspect_ratio']['p05']:.2f}/{refined_stats['aspect_ratio']['p95']:.2f}")
    reference = report.get("legacy_reference")
    if reference:
        LOGGER.info("%s", "-" * 84)
        LOGGER.info("REFERENCE MATCH (legacy crops located by exact template matching)")
        LOGGER.info("  located                     %6d (%d crops not byte-exact)",
                    reference["legacy_crops_located"], reference["legacy_crops_not_byte_exact"])
        for entry in reference["photographs_excluded_from_reference"]:
            LOGGER.info(
                "  EXCLUDED from reference: %s -- %d/%d crops are not sub-images of the raw file "
                "(the raw was re-oriented after cropping); refined found %d seeds there.",
                entry["photograph"], entry["not_byte_exact"], entry["legacy_crops"],
                entry["refined_crops"],
            )
        LOGGER.info("  matched by refined detector %6d  -> recall %.2f%%",
                    reference["matched"], 100 * reference["recall_of_legacy"])
        LOGGER.info("  legacy only (possibly missed)%5d", reference["legacy_only"])
        LOGGER.info("  refined only (newly found)  %6d", reference["refined_only"])
        LOGGER.info("  duplicate pairs: legacy %d, refined %d",
                    reference["legacy_duplicate_pairs"], report["duplicate_detections"]["count"])
    LOGGER.info("%s\nWrote %s", "=" * 84, report_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
