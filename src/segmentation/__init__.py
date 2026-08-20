"""Seed instance segmentation and refined-dataset extraction.

This package turns ``RAW_Samples`` -- 99 photographs of seeds scattered on a
sheet of paper -- into a corpus of one-seed-per-file square crops, together with
the provenance and quality record needed to audit the result.

It is strictly *upstream* of everything else: nothing in ``src/models``,
``src/losses`` or ``src/trainers`` imports it, and it never reads a checkpoint.
The only thing it hands downstream is a directory tree in the same
``root/<seed_type>/<sub_variety>/<IMG_xxxx>_bbox<n>.png`` layout the existing
:class:`~src.datasets.dataset.HierarchicalSeedDataset` already walks, so the
refined corpus is a drop-in replacement selected by ``$SEED_DATA_ROOT``.

Module map
----------

``illumination``
    The photometric model of the scene: the paper's illumination field and the
    support (paper) region. Every later decision is made against these rather
    than against a global threshold, because a global threshold fails on this
    corpus -- Otsu splits the *paper gradient* on photographs whose foreground
    is under ~5 % of the frame.

``detect``
    Foreground scoring and binarisation, connected components, and the
    distance-transform watershed that separates touching seeds.

``instances``
    :class:`~src.segmentation.instances.SeedInstance`: one detection, its shape
    and colour descriptors, its quality verdict, and the square crop policy.

``extract``
    The Hydra entry point that runs the whole corpus and writes the refined
    dataset plus ``manifest.csv`` / ``manifest.json``.

``audit``
    The Hydra entry point that validates a refined dataset: it recovers the
    *exact* bounding boxes of the legacy ``Cropped_Samples`` by template
    matching (the legacy crops are byte-identical sub-images of the raw
    photographs, which is what makes this a reference rather than a guess) and
    reports recall, duplication, fragmentation and coverage against them.

``visualize``
    Per-photograph overlays, rejection galleries and before/after panels.
"""

from src.segmentation.detect import (
    DetectionParams,
    binarise,
    component_instances,
    foreground_score,
    split_touching,
)
from src.segmentation.illumination import (
    SceneModel,
    illumination_field,
    model_scene,
    resolve_downscale,
    support_region,
)
from src.segmentation.instances import (
    REJECTION_REASONS,
    CropPolicy,
    SeedInstance,
    render_crop,
)

__all__ = [
    "REJECTION_REASONS",
    "CropPolicy",
    "DetectionParams",
    "SceneModel",
    "SeedInstance",
    "binarise",
    "component_instances",
    "foreground_score",
    "illumination_field",
    "model_scene",
    "render_crop",
    "resolve_downscale",
    "split_touching",
    "support_region",
]
