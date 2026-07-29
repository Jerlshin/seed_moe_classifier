"""Self-supervised backbone wrappers (paper Section 4)."""

from src.models.backbones.swinv2_dino import DINO, DINOHead, build_dino

__all__ = ["DINO", "DINOHead", "build_dino"]
