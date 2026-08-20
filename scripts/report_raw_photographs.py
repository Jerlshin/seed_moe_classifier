#!/usr/bin/env python
"""Which source photographs exist and were never cropped.

    python scripts/report_raw_photographs.py
    python scripts/report_raw_photographs.py --raw ../Dataset/Hierarchical_SeedData/RAW_Samples
    python scripts/report_raw_photographs.py --json coverage.json

**This script only reports. It never touches the dataset.**

Why it matters more than the crop count suggests. The binding constraint on this
dataset is the number of *scenes*, not the number of crops: within one photograph
89-98 % of crops have a neighbour above cosine 0.95 at 32x32 grey, so the
effective sample count is far below the file count.

On the canonical corpus (``Refined_Samples``, built by ``main.py extract-seeds``)
this reports **3 of 99** uncropped, and all three are excluded on purpose: they
are not trays of seeds. It reported **18** on the hand-curated
``Cropped_Samples``, of which 15 were ordinary trays that had simply never been
cropped -- which is what stage 0 recovered.

It does **not** fix the five single-photograph sub-varieties (Baryard, Browntop,
FingerMillet, PearlMillet, ProsaMillet -- 14.8 % of the crops). Each genuinely
has one raw photograph, so under any photograph-disjoint split they are
unpredictable, identically for every encoder including a random one. The fix
there is a camera, not a splitter.

Two things to do before acting on this report:

1. **Look at the photographs.** They may have been excluded deliberately -- blur,
   exposure, a different tray, or not a tray at all. An out-of-focus photograph
   added to the SSL corpus is worse than nothing, and stage 0's scene gate
   excludes three frames for exactly this reason.
2. **Treat re-cropping as a re-baseline, not an increment.** Every published
   accuracy moves, so it must happen before a phase rather than between arms, and
   the two corpora must stay distinguishable afterwards. They do: stage 1 records
   a corpus SHA-256 in ``summary.json`` and the evaluation cross-checks it, so a
   result produced on the old corpus is identifiable rather than merely stale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import raw_photograph_coverage  # noqa: E402


def default_cropped() -> str:
    """The corpus to audit coverage *of* -- the canonical one unless told otherwise."""
    return os.environ.get(
        "SEED_DATA_ROOT", str(PROJECT_ROOT / "data/Hierarchical_SeedData/Refined_Samples")
    )


def default_raw() -> str:
    """``$SEED_RAW_DATA_ROOT``, else ``RAW_Samples`` beside the cropped tree."""
    override = os.environ.get("SEED_RAW_DATA_ROOT")
    if override:
        return override
    return str(Path(default_cropped()).parent / "RAW_Samples")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", default=default_raw(), help="RAW_Samples tree.")
    parser.add_argument(
        "--cropped", default=default_cropped(),
        help="The corpus tree to audit (default: $SEED_DATA_ROOT, i.e. Refined_Samples).",
    )
    parser.add_argument("--json", default=None, help="Also write the report as JSON.")
    arguments = parser.parse_args()

    coverage = raw_photograph_coverage(arguments.raw, arguments.cropped)
    if not coverage:
        print(
            f"Nothing to report: {arguments.raw} or {arguments.cropped} does not exist.\n"
            "Point --raw at the RAW_Samples tree, or set $SEED_RAW_DATA_ROOT."
        )
        return 1

    print(f"raw     : {coverage['raw_root']}")
    print(f"cropped : {coverage['cropped_root']}")
    print(
        f"\n{coverage['num_used_photographs']} of {coverage['num_raw_photographs']} source "
        f"photographs were cropped; {coverage['num_unused_photographs']} were not.\n"
    )
    print(f"{'sub-variety':<24} {'raw':>5} {'used':>5} {'unused':>7}  unused stems")
    print("-" * 96)
    for name, entry in coverage["per_sub_variety"].items():
        unused = entry["unused"]
        print(
            f"{name:<24} {entry['raw_photographs']:>5} {entry['used']:>5} {len(unused):>7}  "
            + (", ".join(sorted(unused)[:6]) + ("..." if len(unused) > 6 else "") if unused else "")
        )
    print("-" * 96)
    print(
        "\nScene count -- not crop count -- is what a photograph-disjoint protocol can resolve.\n"
        "Inspect the unused photographs before cropping them: an out-of-focus frame added to the\n"
        "SSL corpus is worse than nothing. Re-cropping moves every published accuracy, so do it\n"
        "once, before a phase. The corpus SHA-256 in each run's summary.json keeps the before and\n"
        "after distinguishable."
    )

    if arguments.json:
        Path(arguments.json).write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        print(f"\nWrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
