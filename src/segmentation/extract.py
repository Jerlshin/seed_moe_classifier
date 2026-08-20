"""Build the refined corpus: ``RAW_Samples`` in, one square crop per seed out.

    python -m src.segmentation.extract
    python -m src.segmentation.extract segmentation.crop.margin=0.20
    python -m src.segmentation.extract paths.output_root=/tmp/try segmentation.output.overwrite=true

What it writes
--------------

::

    <output_root>/<seed_type>/<sub_variety>/<IMG_xxxx>_bbox<nnnn>.png
    <report_dir>/manifest.csv          one row per DETECTION, accepted or not
    <report_dir>/manifest.json         the same, as JSON
    <report_dir>/contours.json         boundary polygon per written crop
    <report_dir>/photographs.csv       one row per photograph
    <report_dir>/extraction_summary.json
    <report_dir>/overlays/<sub_variety>__<stem>.png
    <report_dir>/galleries/...

The crop filenames keep the legacy ``_bbox<n>`` convention on purpose. It is
what :func:`src.datasets.dataset.source_image_id` parses to group crops by
source photograph, and that grouping is what the photograph-disjoint split
protocol is built on -- so the refined corpus is a drop-in ``$SEED_DATA_ROOT``
with no change anywhere downstream.

What it never does
------------------

* **It never writes under ``raw_root``.** The raw photographs are the only
  irreplaceable artifact here.
* **It never resamples a crop.** Every emitted pixel is a source pixel. The
  square window is cut, not resized, so the only resize in the whole pipeline is
  the one the augmentation applies -- explicit, configurable and reversible.
* **It never drops a detection silently.** Every connected component that
  survived the size floor gets a manifest row, an outline in the overlay and a
  rejection reason if it was not written.
* **It refuses to write into a populated output root** unless
  ``segmentation.output.overwrite=true``, because a half-overwritten corpus is
  indistinguishable from a complete one by anything except the manifest.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.detect import DetectionParams  # noqa: E402
from src.segmentation.instances import CropPolicy, render_crop  # noqa: E402
from src.segmentation.pipeline import (  # noqa: E402
    PhotographResult,
    SceneGate,
    segment_photograph,
)
from src.segmentation.visualize import (  # noqa: E402
    accepted_gallery,
    overlay_photograph,
    rejection_gallery,
    write,
)

LOGGER = logging.getLogger(__name__)

#: Extensions :func:`find_photographs` accepts. Upper- and lower-case both, and
#: matched case-insensitively, because the corpus ships ``.JPG``.
PHOTOGRAPH_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})


def find_photographs(root: Path) -> list[tuple[Path, str, str, str]]:
    """Every ``root/<seed_type>/<sub_variety>/<photograph>`` file, sorted.

    Sorted so the crop indices a run assigns are a deterministic function of the
    tree, not of filesystem enumeration order: two runs over the same corpus
    produce byte-identical filenames, which is what lets the corpus fingerprint
    mean anything.
    """
    found: list[tuple[Path, str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PHOTOGRAPH_EXTENSIONS:
            continue
        relative = path.relative_to(root).parts
        if len(relative) != 3:
            LOGGER.warning(
                "Skipping %s: expected <seed_type>/<sub_variety>/<file>, got %s levels.",
                path, len(relative),
            )
            continue
        found.append((path, relative[0], relative[1], path.stem))
    return found


def build_params(cfg: DictConfig) -> tuple[DetectionParams, CropPolicy, SceneGate]:
    """Turn the Hydra node into the three dataclasses the pipeline takes."""
    detection = DetectionParams(**OmegaConf.to_container(cfg.segmentation.detection, resolve=True))
    policy = CropPolicy(**OmegaConf.to_container(cfg.segmentation.crop, resolve=True))
    gate = SceneGate(**OmegaConf.to_container(cfg.segmentation.scene, resolve=True))
    return detection, policy, gate


def _manifest_row(
    result: PhotographResult, instance, relative_path: str
) -> dict[str, object]:
    return {
        "seed_type": result.seed_type,
        "sub_variety": result.sub_variety,
        "source_photograph": result.stem,
        "source_path": result.path,
        "source_width": result.width,
        "source_height": result.height,
        "photo_seed_area_median": round(result.seed_area, 1),
        "scene_accepted": result.accepted_scene,
        "relative_path": relative_path,
        **instance.row(),
    }


def extract_photograph(
    path: Path,
    seed_type: str,
    sub_variety: str,
    stem: str,
    cfg: DictConfig,
    detection: DetectionParams,
    policy: CropPolicy,
    gate: SceneGate,
    output_root: Path,
    report_dir: Path,
) -> tuple[PhotographResult, list[dict[str, object]]]:
    """Segment one photograph, write its crops and its figures.

    Crop indices are assigned over the **accepted** instances in detection order
    (which is raster order, top-left to bottom-right), so ``_bbox0007`` always
    names the same seed for a given corpus and configuration.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not decode {path}")

    # The label image is always retained: distractor suppression needs to know
    # which pixels inside a crop window belong to a *different* seed, and that
    # is the label image. The figures reuse it; disabling figures only stops it
    # being drawn, never recomputed.
    keep = bool(cfg.segmentation.figures.enabled)
    result = segment_photograph(
        bgr, detection, policy, gate,
        path=str(path), seed_type=seed_type, sub_variety=sub_variety, stem=stem,
        keep_arrays=True,
    )

    rows: list[dict[str, object]] = []
    destination = output_root / seed_type / sub_variety
    template = str(cfg.segmentation.output.filename_template)
    suffix = str(cfg.segmentation.output.image_format).lstrip(".")

    written = 0
    for instance in result.instances:
        relative = ""
        if instance.accepted:
            name = template.format(stem=stem, index=written) + f".{suffix}"
            crop = render_crop(bgr, result.labels, instance, policy)
            destination.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination / name), crop):
                raise OSError(f"Could not write {destination / name}")
            instance.filename = name
            relative = f"{seed_type}/{sub_variety}/{name}"
            written += 1
        rows.append(_manifest_row(result, instance, relative))

    if keep:
        figures = cfg.segmentation.figures
        tag = f"{sub_variety}__{stem}"
        if figures.per_photograph_overlay:
            write(
                report_dir / "overlays" / f"{tag}.png",
                overlay_photograph(bgr, result, long_side=int(figures.overlay_long_side)),
            )
        if figures.rejection_gallery and result.rejected:
            write(
                report_dir / "galleries" / "rejected" / f"{tag}.png",
                rejection_gallery(bgr, result, per_reason=int(figures.gallery_per_reason)),
            )
        if figures.accepted_gallery and result.accepted:
            write(
                report_dir / "galleries" / "accepted" / f"{tag}.png",
                accepted_gallery(bgr, result),
            )
    # An 8 MP int32 label image per photograph would otherwise accumulate across
    # the whole corpus in the caller's result list.
    result.labels = None
    result.mask = None
    return result, rows


def write_reports(
    report_dir: Path,
    rows: list[dict[str, object]],
    photographs: list[dict[str, object]],
    contours: dict[str, list[tuple[int, int]]],
    summary: dict[str, object],
) -> None:
    """Write the four machine-readable products of an extraction run."""
    import csv

    report_dir.mkdir(parents=True, exist_ok=True)

    # A union of keys, not the first row's: the per-photograph rows carry one
    # `reject_<reason>` column per reason that actually fired, so the set differs
    # between photographs and a DictWriter built from row zero would raise.
    def dump(name: str, records: list[dict[str, object]]) -> None:
        if not records:
            return
        fields: list[str] = []
        seen: set[str] = set()
        for record in records:
            for key in record:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        with (report_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(records)

    dump("manifest.csv", rows)
    dump("photographs.csv", photographs)
    (report_dir / "manifest.json").write_text(
        json.dumps({"detections": rows}, indent=1), encoding="utf-8"
    )
    # Boundary polygons go in their own file. They are the bulk of the bytes --
    # ~50 MB against 1 MB for the detections -- and putting them in
    # `manifest.json` makes the file nobody wants to open the file everybody has
    # to. Written unindented for the same reason.
    if contours:
        (report_dir / "contours.json").write_text(json.dumps(contours), encoding="utf-8")
    (report_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def summarise(
    rows: list[dict[str, object]], photographs: list[dict[str, object]], cfg: DictConfig
) -> dict[str, object]:
    """The numbers a reader wants first, computed from the manifest itself."""
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]

    reasons: dict[str, int] = {}
    for row in rejected:
        for reason in str(row["reasons"]).split("|"):
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1

    sides = np.array([row["window_side"] for row in accepted], dtype=np.float64)
    areas = np.array([row["area"] for row in accepted], dtype=np.float64)
    per_class: dict[str, int] = {}
    for row in accepted:
        key = f"{row['seed_type']}/{row['sub_variety']}"
        per_class[key] = per_class.get(key, 0) + 1

    kept_photographs = [item for item in photographs if item["scene_accepted"]]
    return {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "photographs_found": len(photographs),
        "photographs_used": len(kept_photographs),
        "photographs_rejected": len(photographs) - len(kept_photographs),
        "rejected_photographs": [
            {"path": item["path"], "reasons": item["scene_reasons"],
             "support_fraction": item["support_fraction"],
             "num_accepted": item["num_accepted"]}
            for item in photographs if not item["scene_accepted"]
        ],
        "detections_total": len(rows),
        "crops_written": len(accepted),
        "detections_rejected": len(rejected),
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "crops_from_split_clusters": sum(1 for row in accepted if row["origin"] == "split"),
        "crops_oversized_single": sum(
            1 for row in accepted if row["origin"] == "oversized_single"
        ),
        "crops_with_distractor_suppressed": sum(
            1 for row in accepted if float(row["distractor_fraction"]) > 0
        ),
        "crops_with_shifted_window": sum(1 for row in accepted if row["window_shifted"]),
        "crop_side_px": {
            "min": int(sides.min()) if sides.size else 0,
            "p05": float(np.percentile(sides, 5)) if sides.size else 0.0,
            "median": float(np.median(sides)) if sides.size else 0.0,
            "p95": float(np.percentile(sides, 95)) if sides.size else 0.0,
            "max": int(sides.max()) if sides.size else 0,
            "mean": float(sides.mean()) if sides.size else 0.0,
        },
        "seed_area_px": {
            "median": float(np.median(areas)) if areas.size else 0.0,
            "p05": float(np.percentile(areas, 5)) if areas.size else 0.0,
            "p95": float(np.percentile(areas, 95)) if areas.size else 0.0,
        },
        "crops_per_class": dict(sorted(per_class.items())),
        "classes": len(per_class),
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="segmentation")
def main(cfg: DictConfig) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raw_root = Path(cfg.paths.raw_root).expanduser()
    output_root = Path(cfg.paths.output_root).expanduser()
    report_dir = Path(cfg.paths.report_dir).expanduser()

    if not raw_root.exists():
        LOGGER.error(
            "Raw photographs not found at %s. Set $SEED_RAW_DATA_ROOT or pass "
            "paths.raw_root=<path>.", raw_root,
        )
        return 1

    existing = (
        [item for item in output_root.rglob("*") if item.is_file()]
        if output_root.exists()
        else []
    )
    if existing and not bool(cfg.segmentation.output.overwrite):
        LOGGER.error(
            "%s already holds %s files. Extraction refuses to write into a populated root "
            "because a partial overwrite is indistinguishable from a complete corpus. "
            "Delete it, choose another paths.output_root, or pass "
            "segmentation.output.overwrite=true.", output_root, len(existing),
        )
        return 1
    if existing:
        LOGGER.warning("Overwriting %s existing files under %s.", len(existing), output_root)
        for item in existing:
            item.unlink()

    detection, policy, gate = build_params(cfg)
    photographs = find_photographs(raw_root)
    if not photographs:
        LOGGER.error("No photographs under %s.", raw_root)
        return 1

    LOGGER.info(
        "Extracting from %s photographs under %s -> %s", len(photographs), raw_root, output_root
    )
    rows: list[dict[str, object]] = []
    photo_rows: list[dict[str, object]] = []
    contours: dict[str, list[tuple[int, int]]] = {}
    started = time.time()

    for number, (path, seed_type, sub_variety, stem) in enumerate(photographs, start=1):
        result, photo_manifest = extract_photograph(
            path, seed_type, sub_variety, stem, cfg, detection, policy, gate,
            output_root, report_dir,
        )
        rows.extend(photo_manifest)
        photo_rows.append(result.summary())
        if bool(cfg.segmentation.output.write_contours):
            for instance in result.accepted:
                contours[f"{seed_type}/{sub_variety}/{instance.filename}"] = instance.contour
        LOGGER.info(
            "[%3d/%3d] %-22s %-14s %5d detections -> %4d crops%s",
            number, len(photographs), sub_variety, stem,
            len(result.instances), len(result.accepted),
            "" if result.accepted_scene else f"   SCENE REJECTED: {','.join(result.scene_reasons)}",
        )

    summary = summarise(rows, photo_rows, cfg)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["output_root"] = str(output_root)
    summary["raw_root"] = str(raw_root)

    # The corpus fingerprint is what makes a result traceable to the corpus that
    # produced it. Recorded here so the extraction run itself, not only a later
    # training run, carries the identity of what it built.
    try:
        from src.datasets.dataset import corpus_fingerprint

        summary["corpus_fingerprint"] = corpus_fingerprint(output_root)
    except Exception as error:  # pragma: no cover - fingerprint is a report, not a gate
        LOGGER.warning("Could not fingerprint the new corpus: %s", error)

    write_reports(report_dir, rows, photo_rows, contours, summary)

    LOGGER.info(
        "\n%s\nphotographs: %s found, %s used, %s rejected\n"
        "detections : %s total, %s written, %s rejected\n"
        "crop side  : median %.0f px (p5 %.0f, p95 %.0f, max %d)\n"
        "split from touching clusters: %s | distractors inpainted: %s\n"
        "reports    : %s\n%s",
        "=" * 78,
        summary["photographs_found"], summary["photographs_used"], summary["photographs_rejected"],
        summary["detections_total"], summary["crops_written"], summary["detections_rejected"],
        summary["crop_side_px"]["median"], summary["crop_side_px"]["p05"],
        summary["crop_side_px"]["p95"], summary["crop_side_px"]["max"],
        summary["crops_from_split_clusters"], summary["crops_with_distractor_suppressed"],
        report_dir, "=" * 78,
    )
    for entry in summary["rejected_photographs"]:
        LOGGER.warning(
            "Photograph excluded: %s (%s, support %.2f, %s plausible seeds)",
            entry["path"], entry["reasons"], entry["support_fraction"], entry["num_accepted"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
