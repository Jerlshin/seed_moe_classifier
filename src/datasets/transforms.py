"""Multi-crop DINO augmentation and supervised transforms (paper Sections 4, 6.1).

The paper's augmentation recipe, verbatim from Section 6.1:

* **2 global crops**, scale ``(0.4, 1.0)``, both resized to the backbone's input
  resolution.

  - Global crop 1: random resized crop, colour jitter, Gaussian blur ``p = 1.0``,
    normalize.
  - Global crop 2: the same, but Gaussian blur ``p = 0.1`` and solarization
    ``p = 0.2``.

* **4 local crops**, scale ``(0.05, 0.4)``, bicubic interpolation, colour jitter,
  Gaussian blur ``p = 0.5``, normalize.

* Colour jitter magnitudes: brightness ``+/-0.4``, contrast ``+/-0.4``,
  saturation ``+/-0.2``, hue ``+/-0.1``.

The asymmetry between the two global crops is what makes the teacher's two views
genuinely different, which is the signal the cross-view DINO loss consumes.

One implementation note: SwinV2's shifted windows require a fixed input size, so
``resize_local_to_global`` scales local crops back up to ``image_size`` after
cropping. The crop *scale* is what carries the local/global distinction, not the
final tensor size.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur:
    """Randomly apply a Gaussian blur with a uniformly sampled radius."""

    def __init__(self, probability: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.probability = float(probability)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        radius = random.uniform(self.radius_min, self.radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.probability})"


class Solarization:
    """Randomly invert pixels above a threshold (paper: applied to global crop 2)."""

    def __init__(self, probability: float = 0.2, threshold: int = 128):
        self.probability = float(probability)
        self.threshold = int(threshold)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        return ImageOps.solarize(image, threshold=self.threshold)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.probability})"


class DataAugmentationDINO:
    """Produce ``2 + local_crops_number`` augmented views of one image.

    Args:
        image_size: Backbone input resolution; global crops land at this size.
        local_crop_size: Crop size for local views before optional upsizing.
        global_crops_scale: Area range for global crops (paper: 0.4-1.0).
        local_crops_scale: Area range for local crops (paper: 0.05-0.4).
        local_crops_number: Local views per image (paper: 4).
        horizontal_flip_prob: Probability of a horizontal flip.
        color_jitter_prob: Probability of applying colour jitter at all.
        color_jitter_brightness/contrast/saturation/hue: Jitter magnitudes.
        grayscale_prob: Probability of a grayscale conversion.
        global_blur_prob_1: Blur probability on global crop 1 (paper: 1.0).
        global_blur_prob_2: Blur probability on global crop 2 (paper: 0.1).
        local_blur_prob: Blur probability on local crops (paper: 0.5).
        solarization_prob: Solarization probability on global crop 2 (paper: 0.2).
        resize_local_to_global: Resize local crops back up to ``image_size``.
        normalize_mean / normalize_std: Channel normalisation statistics.
    """

    def __init__(
        self,
        image_size: int,
        local_crop_size: int,
        global_crops_scale: Sequence[float] = (0.4, 1.0),
        local_crops_scale: Sequence[float] = (0.05, 0.4),
        local_crops_number: int = 4,
        horizontal_flip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
        color_jitter_brightness: float = 0.4,
        color_jitter_contrast: float = 0.4,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
        grayscale_prob: float = 0.2,
        global_blur_prob_1: float = 1.0,
        global_blur_prob_2: float = 0.1,
        local_blur_prob: float = 0.5,
        solarization_prob: float = 0.2,
        resize_local_to_global: bool = True,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        if local_crops_number < 0:
            raise ValueError(f"local_crops_number must be >= 0, got {local_crops_number}")

        interpolation = T.InterpolationMode.BICUBIC
        flip_and_color_jitter = T.Compose(
            [
                T.RandomHorizontalFlip(p=horizontal_flip_prob),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=color_jitter_brightness,
                            contrast=color_jitter_contrast,
                            saturation=color_jitter_saturation,
                            hue=color_jitter_hue,
                        )
                    ],
                    p=color_jitter_prob,
                ),
                T.RandomGrayscale(p=grayscale_prob),
            ]
        )
        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(tuple(normalize_mean), tuple(normalize_std)),
            ]
        )

        # Un-augmented view, kept for visualisation and attention overlays.
        self.original_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=interpolation),
                T.ToTensor(),
            ]
        )
        self.global_transform_1 = T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=tuple(global_crops_scale), interpolation=interpolation),
                flip_and_color_jitter,
                GaussianBlur(global_blur_prob_1),
                normalize,
            ]
        )
        self.global_transform_2 = T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=tuple(global_crops_scale), interpolation=interpolation),
                flip_and_color_jitter,
                GaussianBlur(global_blur_prob_2),
                Solarization(solarization_prob),
                normalize,
            ]
        )

        local_steps: list[Any] = [
            T.RandomResizedCrop(local_crop_size, scale=tuple(local_crops_scale), interpolation=interpolation),
            flip_and_color_jitter,
            GaussianBlur(local_blur_prob),
            normalize,
        ]
        if resize_local_to_global:
            # SwinV2 needs a fixed input resolution; the local/global distinction
            # is carried by the crop scale, not by the final tensor size.
            local_steps.append(T.Resize((image_size, image_size), interpolation=interpolation))

        self.local_crops_number = int(local_crops_number)
        self.local_transform = T.Compose(local_steps)

    @property
    def num_crops(self) -> int:
        """Total views per image: 2 global + ``local_crops_number`` local."""
        return 2 + self.local_crops_number

    def __call__(self, image: Image.Image):
        """Return ``(original_tensor, [global_1, global_2, *locals])``."""
        image = image.convert("RGB")
        crops = [self.global_transform_1(image), self.global_transform_2(image)]
        crops.extend(self.local_transform(image) for _ in range(self.local_crops_number))
        return self.original_transform(image), crops


def get_dino_transforms(image_size: int, local_crop_size: int, augmentation_cfg: Any | None = None):
    """Build :class:`DataAugmentationDINO` from a ``data.augmentation`` config node."""
    params = dict(augmentation_cfg or {})
    # `local_crop_size` lives on the data node, not inside `augmentation`.
    params.pop("local_crop_size", None)
    return DataAugmentationDINO(image_size=image_size, local_crop_size=local_crop_size, **params)


def get_supervised_transforms(
    image_size: int,
    train: bool = True,
    normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
    normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    horizontal_flip_prob: float = 0.0,
):
    """Deterministic resize + normalize used by both finetuning and evaluation.

    Stage 2 keeps augmentation minimal on purpose: the representation is already
    invariant from DINO pretraining, and the fine-grained cues that separate
    sub-varieties are exactly the ones heavy augmentation destroys.
    """
    steps: list[Any] = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
    ]
    if train and horizontal_flip_prob > 0:
        steps.append(T.RandomHorizontalFlip(p=horizontal_flip_prob))
    steps.extend(
        [
            T.ToTensor(),
            T.Normalize(tuple(normalize_mean), tuple(normalize_std)),
        ]
    )
    return T.Compose(steps)
