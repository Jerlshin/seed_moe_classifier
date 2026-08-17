"""Shared fixtures and the paper's architectural constants.

Nothing here touches the network or the real dataset: the synthetic image tree
mirrors the paper's 4-type / 27-sub-variety hierarchy at a size that builds in
milliseconds, and the feature extractor is stubbed so tests never download a
backbone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paper constants. A test that contradicts one of these is reporting a
# divergence from the paper, not a flaky assertion.
# ---------------------------------------------------------------------------
PAPER_EMBED_DIM = 384          # Eq. 4, z in R^384
PAPER_NUM_SEED_TYPES = 4       # Section 5.1: rice, millet, amaranthus, mustard
PAPER_NUM_SUB_VARIETIES = 27   # Section 3
PAPER_NUM_EXPERTS = 6          # Section 5.2

# Routing width. The submitted manuscript activated the top 4 of 6 experts;
# the revision routes 2. Both are kept as named constants so a test can state
# which one it means -- test_efficiency compares them directly, and a test that
# asserted a bare "4" would silently become a claim about the wrong paper.
SUBMITTED_TOP_K = 4
REVISED_TOP_K = 2

# Token grid emitted by swinv2_*_window16_256 at 256 px. Grid routing is the
# revision default; `pooled` reproduces the submitted architecture.
PAPER_TOKEN_GRID = 64

PAPER_GLOBAL_CROPS = 2         # Table 1
PAPER_LOCAL_CROPS = 4          # Table 1
PAPER_CLIP_GRAD = 3.0

# Stage-1 constants come in pairs now: what Table 1 states, and what the revision
# uses. A test asserting a bare number would silently become a claim about
# whichever of the two the reader assumed.
#
# The submitted values put both collapse guards in the collapsing direction at
# once -- sharpening ~2x stronger than DINO's, and a 65,536-dimensional centering
# vector estimated from ~320 effective samples. See architecture/02.
SUBMITTED_TEACHER_MOMENTUM = 0.996
REVISED_TEACHER_MOMENTUM_FINAL = 1.0
SUBMITTED_WARMUP_TEACHER_TEMP = 0.02
SUBMITTED_TEACHER_TEMP = 0.04
REVISED_WARMUP_TEACHER_TEMP = 0.04
REVISED_TEACHER_TEMP = 0.07
SUBMITTED_WARMUP_EPOCHS = 5
REVISED_WARMUP_EPOCHS = 30
SUBMITTED_DINO_OUT_DIM = 65536
#: The first revision's prototype count. Kept so a test can say which revision
#: it means: 8,192 was chosen against a batch of 16, and the current 2,048 is
#: the value that follows from raising the physical batch and reading Sinkhorn's
#: per-prototype evidence (B_teacher / K) as the quantity that matters.
FIRST_REVISION_DINO_OUT_DIM = 8192
REVISED_DINO_OUT_DIM = 2048
SUBMITTED_CENTER_MOMENTUM = 0.9
REVISED_CENTER_MOMENTUM = 0.99

# Stage-1 backbone and recipe, second revision. The submitted trunk was
# SwinV2-Base at random initialisation for 300 epochs; the revision starts from
# ImageNet-1k on SwinV2-Tiny for 100. All four numbers below were MEASURED from
# timm rather than quoted, and `test_models.py` re-measures them.
SUBMITTED_BACKBONE = "swinv2_base_window16_256"
REVISED_BACKBONE = "swinv2_tiny_window16_256"
SUBMITTED_BACKBONE_FEATURE_DIM = 1024
REVISED_BACKBONE_FEATURE_DIM = 768
REVISED_BACKBONE_PARAMS_M = 27.58      # +-0.01, swinv2_tiny_window16_256
REVISED_BACKBONE_GFLOPS_PER_VIEW = 13.32   # forward, 1 view at 256 px

# The trunk `conf/model/backbone/swinv2.yaml` actually selects, and the one the
# published 100-epoch checkpoint was trained with: SwinV2-**Small**, a third
# constant rather than an edit to REVISED_BACKBONE, because the Tiny numbers above
# are still what several tests measure and a test asserting a bare name must be
# able to say which trunk it means.
#
# Both figures are MEASURED: the params from timm, the GFLOPs from
# FlopCounterMode, and the stage-1 run's own budget report printed 25.57
# GFLOPs/view for backbone + head against the 25.56 measured here for the
# backbone alone.
SHIPPED_BACKBONE = "swinv2_small_window16_256"
SHIPPED_BACKBONE_FEATURE_DIM = 768
SHIPPED_BACKBONE_PARAMS_M = 48.96
SHIPPED_BACKBONE_GFLOPS_PER_VIEW = 25.56
SUBMITTED_EPOCHS = 300
REVISED_EPOCHS = 100
REVISED_LR_WARMUP_EPOCHS = 10
REVISED_DROP_PATH_RATE = 0.1

# Learning rate. Section 6.1's 0.0005 is DINO's rate at ITS reference batch of
# 256; the revision derives the run's rate from the effective batch by the
# linear scaling rule, which at 32 gives 6.25e-05.
LR_BASE = 0.0005
LR_REFERENCE_BATCH = 256
REVISED_PHYSICAL_BATCH = 32
REVISED_EFFECTIVE_BATCH = 32
REVISED_LEARNING_RATE = LR_BASE * REVISED_EFFECTIVE_BATCH / LR_REFERENCE_BATCH

# DINO projection head, resized with the trunk: 768 -> 1024 -> 1024 -> 256 -> 2048.
REVISED_DINO_HIDDEN_DIM = 1024
REVISED_DINO_BOTTLENECK_DIM = 256
REVISED_DINO_HEAD_LAYERS = 3

# AdaCos fixed scale for C = 27: sqrt(2) * ln 26. The submitted 30.0 is
# ArcFace's face-recognition value and is 6.5x too large here.
ADACOS_SCALE_27 = 4.6076
SUBMITTED_ARCFACE_SCALE = 30.0

# Dataset provenance, measured from the real tree under Cropped_Samples. These
# decide what any accuracy number can mean, so they are constants like the rest.
DATASET_NUM_SOURCE_PHOTOGRAPHS = 81
DATASET_NUM_CROPS = 9357
DATASET_SINGLE_SOURCE_SUBVARIETIES = 5

# 13 rice + 8 millet + 3 + 3 = 27, matching Section 3.
SUBVARIETY_COUNTS = {"Millet": 8, "Mustard": 3, "Rice": 13, "Seasame": 3}


@pytest.fixture(scope="session")
def subvariety_to_seed_type() -> list[int]:
    """Parent seed type per sub-variety index, in sorted-directory order."""
    mapping: list[int] = []
    for seed_index, name in enumerate(sorted(SUBVARIETY_COUNTS)):
        mapping.extend([seed_index] * SUBVARIETY_COUNTS[name])
    assert len(mapping) == PAPER_NUM_SUB_VARIETIES
    return mapping


# Enough samples per sub-variety that a stratified 27-class train/val/test split
# still leaves at least two members of every class in every partition.
IMAGES_PER_SUBVARIETY = 12


@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory) -> Path:
    """A ``root/<seed_type>/<sub_variety>/*.png`` tree matching the paper's shape."""
    root = tmp_path_factory.mktemp("seed_data")
    for seed_type, count in SUBVARIETY_COUNTS.items():
        for sub_index in range(count):
            directory = root / seed_type / f"{seed_type}_sub{sub_index:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            for image_index in range(IMAGES_PER_SUBVARIETY):
                image = Image.fromarray(
                    (torch.rand(16, 16, 3) * 255).to(torch.uint8).numpy(), mode="RGB"
                )
                image.save(directory / f"img{image_index}.png")
    return root


def build_head(token_mode: str = "pooled", **overrides):
    """The paper's head at its stated dimensions, with an identity input projection."""
    from src.models.builder import HierarchicalSeedClassifier

    torch.manual_seed(0)
    kwargs = {
        "feature_dim": PAPER_EMBED_DIM,
        "embed_dim": PAPER_EMBED_DIM,
        "num_seed_types": PAPER_NUM_SEED_TYPES,
        "num_sub_varieties": PAPER_NUM_SUB_VARIETIES,
        "num_experts": PAPER_NUM_EXPERTS,
        "top_k": REVISED_TOP_K,
        "moe_hidden_dim": 64,
        "num_heads": 4,
        "dropout_rate": 0.0,
        "token_mode": token_mode,
    }
    kwargs.update(overrides)
    return HierarchicalSeedClassifier(**kwargs)


@pytest.fixture
def hierarchical_model():
    """Pooled-mode head: one token per image, as the submitted architecture had.

    Pooled is the fixture default because most invariants are about the
    coarse-to-fine dataflow rather than routing granularity, and a ``[batch,
    384]`` output keeps those assertions readable. ``grid_model`` covers what
    only the token grid can express.
    """
    return build_head("pooled")


@pytest.fixture
def grid_model():
    """Grid-mode head: 64 tokens per image, so the attention is non-degenerate."""
    return build_head("grid")


@pytest.fixture
def combined_loss(subvariety_to_seed_type):
    """The full combined objective wired to the paper's hierarchy."""
    from src.losses.hierarchical import CombinedHierarchicalLoss

    return CombinedHierarchicalLoss(
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
        embed_dim=PAPER_EMBED_DIM,
        num_experts=PAPER_NUM_EXPERTS,
    )


@pytest.fixture
def batch():
    """A small batch of pooled features and hierarchical labels."""
    torch.manual_seed(1)
    size = 12
    return {
        "features": torch.randn(size, PAPER_EMBED_DIM),
        "seed_labels": torch.randint(0, PAPER_NUM_SEED_TYPES, (size,)),
        "sub_labels": torch.randint(0, PAPER_NUM_SUB_VARIETIES, (size,)),
        "size": size,
    }


@pytest.fixture
def token_batch():
    """A small batch of token grids, for the grid-routing invariants."""
    torch.manual_seed(1)
    size, tokens = 6, 16
    return {
        "features": torch.randn(size, tokens, PAPER_EMBED_DIM),
        "seed_labels": torch.randint(0, PAPER_NUM_SEED_TYPES, (size,)),
        "sub_labels": torch.randint(0, PAPER_NUM_SUB_VARIETIES, (size,)),
        "size": size,
        "tokens": tokens,
    }


class DummyEncoder(nn.Module):
    """Stand-in for the DINOv2-SwinV2 encoder: images -> ``[batch, 384]``.

    Keeps integration tests free of network access and of multi-hundred-megabyte
    weight downloads, while preserving the contract the head depends on -- the
    same one :class:`~src.models.builder.DinoV2SwinV2Encoder` guarantees, namely
    a 2-D output of width ``embed_dim``.
    """

    load_report = None

    def __init__(self, embed_dim: int = PAPER_EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim
        self.frozen = True
        self.project = nn.Linear(3, embed_dim)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pooled = images.mean(dim=(2, 3))  # [batch, 3]
            return self.project(pooled)


#: Retained so older references keep working.
DummyFeatureExtractor = DummyEncoder


@pytest.fixture
def dummy_encoder() -> DummyEncoder:
    torch.manual_seed(2)
    return DummyEncoder()


@pytest.fixture
def conf_dir() -> str:
    """Absolute path to the Hydra config tree."""
    return os.path.join(str(PROJECT_ROOT), "conf")
