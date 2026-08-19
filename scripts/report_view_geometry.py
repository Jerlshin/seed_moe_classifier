#!/usr/bin/env python
"""What each DINO view is actually built from -- before spending a GPU-hour on it.

    python scripts/report_view_geometry.py
    python scripts/report_view_geometry.py --data seed_crops_v2
    python scripts/report_view_geometry.py --compare hierarchical_seeds seed_crops_v2
    python scripts/report_view_geometry.py --csv outputs/reports/view_geometry.csv

``RandomResizedCrop``'s ``scale`` is a fraction of the **source area**, so a
configuration does not say how much of a seed a view contains -- only the product
of the scale range and the source-size distribution does. On this corpus the
source is one seed at a median 52 x 51 px, and the submitted recipe builds each
local view from a median **598 native pixels** rendered into 65,536 output
pixels, with 8 of the 10 cross-view terms in Eq. 1 anchored on such a view.

This reads the real file headers (~2 s for 9,357 files, no decode) and drives
torchvision's own ``RandomResizedCrop.get_params``, so the numbers are what the
dataloader will produce rather than a model of it. That matters for one result
in particular: ``get_params`` retries the (area, aspect) draw ten times and then
returns a **deterministic centre crop**, and on a corpus that is 96.6 %
non-square, raising the scale floor pushes it into that fallback. The rate is
reported per view family and is the reason ``crop_ratio`` is a config key.

Reads nothing but the dataset and the config, writes nothing but the CSV it is
asked for, and never touches a checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: The columns worth putting side by side when comparing two policies. Ordered
#: by what a reader should look at first.
COMPARISON_ROWS = (
    ("native px p5", "native_pixels_p5", "{:.0f}"),
    ("native px median", "native_pixels_p50", "{:.0f}"),
    ("native px p95", "native_pixels_p95", "{:.0f}"),
    ("native side median", "native_side_median", "{:.1f} px"),
    ("upsample median", "upsample_factor_median", "{:.1f}x"),
    ("upsample p95", "upsample_factor_p95", "{:.1f}x"),
    ("real content median", "real_content_fraction_median", "{:.2%}"),
    ("deterministic fallback", "deterministic_fallback_rate", "{:.1%}"),
)


def load_transform(data_group: str, overrides: list[str]):
    """Compose ``conf/config.yaml`` with one ``data`` group and build its transform."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src.datasets.transforms import get_dino_transforms

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"data={data_group}", *overrides],
        )
    transform = get_dino_transforms(
        int(cfg.data.image_size),
        int(cfg.data.local_crop_size),
        cfg.data.augmentation,
        return_original=False,
    )
    return cfg, transform, OmegaConf.to_container(cfg.data.augmentation, resolve=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        default="hierarchical_seeds",
        help="Config group under conf/data/ to measure (default: hierarchical_seeds).",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="GROUP",
        help="Measure several data groups and print them side by side.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Dataset root. Defaults to $SEED_DATA_ROOT / the config's own value.",
    )
    parser.add_argument("--samples", type=int, default=20000, help="Monte-Carlo draws per family.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default=None, help="Write the full report to this CSV.")
    parser.add_argument(
        "overrides", nargs="*", default=[], help="Extra Hydra overrides, e.g. data.augmentation.min_native_pixels=900"
    )
    args = parser.parse_args()

    from src.datasets.dataset import image_sizes
    from src.datasets.transforms import view_geometry_report

    groups = args.compare or [args.data]
    reports: dict[str, dict] = {}
    sizes: list[tuple[int, int]] = []

    for group in groups:
        cfg, transform, policy = load_transform(group, list(args.overrides))
        root = args.root or os.environ.get("SEED_DATA_ROOT") or str(cfg.data.root_path)
        if not sizes:
            paths = sorted(
                path
                for path in Path(root).rglob("*")
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if not paths:
                print(f"No images found under {root}.", file=sys.stderr)
                return 1
            sizes = image_sizes(paths)
            source = view_geometry_report(transform, sizes, samples=8, seed=args.seed)["source"]
            print(f"Corpus: {root}")
            print(
                f"  {int(source['count'])} images | median {source['width_median']:.0f} x "
                f"{source['height_median']:.0f} px | area p5/p50/p95 = "
                f"{source['area_p5']:.0f}/{source['area_p50']:.0f}/{source['area_p95']:.0f} px^2"
            )
            print(
                f"  {source['square_fraction']:.1%} are square | aspect p5/p95 = "
                f"{source['aspect_ratio_p5']:.2f}/{source['aspect_ratio_p95']:.2f} | "
                f"{source['both_sides_under_64_fraction']:.1%} have both sides under 64 px"
            )
            print()
        reports[group] = view_geometry_report(
            transform, sizes, samples=int(args.samples), seed=int(args.seed)
        )

    for family in ("global", "local"):
        print(f"=== {family} views " + "=" * (56 - len(family)))
        header = f"{'':26s}" + "".join(f"{group:>22s}" for group in groups)
        print(header)
        policy_rows = [
            ("scale", lambda block: f"({block['scale_low']:.2f}, {block['scale_high']:.2f})"),
            ("aspect ratio", lambda block: f"({block['ratio_low']:.2f}, {block['ratio_high']:.2f})"),
            ("native px floor", lambda block: f"{block['min_native_pixels']:.0f}"),
        ]
        for label, render in policy_rows:
            print(f"{label:26s}" + "".join(f"{render(reports[g][family]):>22s}" for g in groups))
        for label, key, fmt in COMPARISON_ROWS:
            print(
                f"{label:26s}"
                + "".join(f"{fmt.format(reports[g][family][key]):>22s}" for g in groups)
            )
        print()

    first = reports[groups[0]]
    print(
        f"{first['local_anchored_loss_term_fraction']:.0%} of Eq. 1's cross-view terms are "
        "anchored on a local view."
    )
    print(
        "A view's information content is its native pixel count, not its tensor size: every "
        "view here is an UPSAMPLE, so the resize adds no information to any of them."
    )
    for group in groups:
        fallback = reports[group]["global"]["deterministic_fallback_rate"]
        if fallback > 0.15:
            print(
                f"WARNING [{group}]: {fallback:.1%} of global draws return torchvision's "
                "deterministic centre crop, so that share of the views carries no crop "
                "randomness. Widen data.augmentation.crop_ratio."
            )

    if args.csv:
        import csv as csv_module

        destination = Path(args.csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"data_group": group, "family": family, **reports[group][family]}
            for group in groups
            for family in ("global", "local")
        ]
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv_module.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
