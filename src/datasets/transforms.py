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

Local crops and a known confound
--------------------------------

SwinV2's shifted windows require a fixed input size, so ``resize_local_to_global``
scales local crops back up to ``image_size`` after cropping. The crop *scale* is
what carries the local/global distinction, not the final tensor size.

This is **not** removable by ``dynamic_img_size=True``. timm accepts the flag for
SwinV2 but the attention still asserts the native resolution: forwarding a 101 px
batch through ``swinv2_*_window16_256`` raises ``AssertionError: Input height
(101) doesn't match model (256)``. ``tests/test_models.py`` pins that, so the
constraint is measured rather than assumed.

The consequence has to be stated rather than worked around: a 101 px crop
upsampled 2.53x to 256 carries a systematic low-pass signature that global crops
do not, so the student can tell local from global views by blur alone -- a
shortcut that partially substitutes for the local-to-global correspondence the
multi-crop objective exists to teach. The per-view blur probabilities (global-1:
1.0, global-2: 0.1, local: 0.5) aggravate it. ``match_view_lowpass`` mitigates it
by putting the *global* crops through the same downsample-then-upsample cycle, so
the artefact carries no discriminative signal; it is off by default because it
changes the submitted recipe, and it is the honest fix if the shortcut turns out
to matter.
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
        match_view_lowpass: Put global crops through the same
            downsample-to-``local_crop_size``-then-upsample cycle the local crops
            undergo, so the resolution artefact stops being a local/global cue.
            See the module docstring.
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
        match_view_lowpass: bool = False,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        if local_crops_number < 0:
            raise ValueError(f"local_crops_number must be >= 0, got {local_crops_number}")

        interpolation = T.InterpolationMode.BICUBIC
        # Applied to the *global* views, reproducing the resolution artefact the
        # local views necessarily carry (see the module docstring).
        lowpass: list[Any] = (
            [
                T.Resize((local_crop_size, local_crop_size), interpolation=interpolation),
                T.Resize((image_size, image_size), interpolation=interpolation),
            ]
            if match_view_lowpass and resize_local_to_global
            else []
        )
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
                *lowpass,
                normalize,
            ]
        )
        self.global_transform_2 = T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=tuple(global_crops_scale), interpolation=interpolation),
                flip_and_color_jitter,
                GaussianBlur(global_blur_prob_2),
                Solarization(solarization_prob),
                *lowpass,
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

    @property
    def view_ids(self) -> list[int]:
        """Identifier of each view the student sees, in emission order.

        The loss skips same-view (teacher, student) pairs. Matching them by
        *position* is only correct while the student's first two views are the
        two globals in the teacher's order -- an invariant nothing enforced, and
        one that would silently skip a global-local pair and include a same-view
        pair if this method's ordering ever changed. Passing identifiers makes
        the pairing explicit instead.
        """
        return list(range(self.num_crops))

    @property
    def global_view_ids(self) -> list[int]:
        """Identifiers of the views the teacher sees."""
        return [0, 1]

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
    horizontal_flip_prob: float = 0.5,
    random_resized_crop_scale: Sequence[float] | None = (0.8, 1.0),
    vertical_flip_prob: float = 0.0,
    rotation_degrees: float = 0.0,
):
    """Stage-2 transforms for finetuning (``train=True``) and evaluation.

    **The resize is an explicit ``(H, W)`` pair, and that is load-bearing.**
    ``T.Resize(int)`` resizes the shorter side and preserves aspect ratio; only
    3.4 % of the crops under ``Cropped_Samples`` are square (aspect ratios span
    0.17 to 3.48), so the integer form would emit variable-width tensors and
    ``default_collate`` would raise on the first mixed batch. Passing the tuple
    squashes every crop to a square, which distorts aspect ratio -- a real but
    deliberate trade, and the honest alternative (pad-to-square) is available by
    editing this one call site.

    Note also that the crops are **small**: median 52x51 px, and 100 % of them
    have both sides under 256. Every image is therefore upsampled ~5x to reach
    the backbone's window resolution, so "fine-grained texture" here means
    texture the sensor resolved at ~50 px, not texture recovered by the resize.

    Augmentation, revised
    ---------------------

    The submitted default was ``horizontal_flip_prob = 0.0``, i.e. stage-2
    training saw each image exactly once per epoch, deterministically. The stated
    rationale -- that the representation is already invariant from stage 1 -- is
    a good argument against *heavy* augmentation but not against *any*: SSL
    invariance of the frozen **encoder** says nothing about the sample efficiency
    of the **head**, which is where ~9 M freshly-initialised parameters are fit
    from ~7.5 k training images. The revision defaults to a flip plus a mild
    ``RandomResizedCrop(scale=(0.8, 1.0))``, which is standard for exactly this
    setting and does not destroy fine texture. Set
    ``random_resized_crop_scale=null`` and ``horizontal_flip_prob=0.0`` to
    reproduce the submitted configuration; ``scripts/run_ablations.py`` runs that
    as the ``wo_stage2_augmentation`` control, because "does stage-2 augmentation
    help when the encoder is frozen?" is a legitimate question whose answer
    belongs in the paper rather than in a config default.

    Args:
        image_size: Target square resolution.
        train: Apply the stochastic steps. Evaluation is always deterministic.
        normalize_mean / normalize_std: Channel normalisation statistics.
        horizontal_flip_prob: Horizontal flip probability.
        random_resized_crop_scale: Area range for the training crop, or ``None``
            for a plain resize.
        vertical_flip_prob: Vertical flip probability. Seeds photographed on a
            tray have no canonical up, so this is safe here; it is off by default
            only because the submitted pipeline did not use it.
        rotation_degrees: Random rotation range in degrees, ``0`` to disable.
    """
    interpolation = T.InterpolationMode.BICUBIC
    steps: list[Any] = []

    if train and random_resized_crop_scale is not None:
        steps.append(
            T.RandomResizedCrop(
                (image_size, image_size),
                scale=tuple(random_resized_crop_scale),
                interpolation=interpolation,
            )
        )
    else:
        steps.append(T.Resize((image_size, image_size), interpolation=interpolation))

    if train:
        if horizontal_flip_prob > 0:
            steps.append(T.RandomHorizontalFlip(p=horizontal_flip_prob))
        if vertical_flip_prob > 0:
            steps.append(T.RandomVerticalFlip(p=vertical_flip_prob))
        if rotation_degrees > 0:
            steps.append(T.RandomRotation(degrees=float(rotation_degrees), interpolation=interpolation))

    steps.extend(
        [
            T.ToTensor(),
            T.Normalize(tuple(normalize_mean), tuple(normalize_std)),
        ]
    )
    return T.Compose(steps)
