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

What the source images actually are, and why the view recipe was redesigned
--------------------------------------------------------------------------

``RandomResizedCrop`` samples a fraction of the **source image's** area. On
ImageNet the source is a whole scene and a 5 % crop is still a recognisable
object part. Here the source is already a tight crop of *one seed* at a median
52x51 px, so the same recipe is not sampling a part of a scene -- it is
shredding the object.

Measured over all 9,357 crops (``scripts/report_view_geometry.py`` reproduces
this, and ``STAGE1_CHANGES.md`` 0.5 measured it independently):

===================================== ============================= =============
view recipe                           native px behind it p5/50/95  upsample to 256
===================================== ============================= =============
global ``scale=(0.40, 1.00)``         648 / 1,845 / 5,680           6.0x median
local  ``scale=(0.05, 0.40)``         132 / **598** / 2,585         **10.5x**
local  ``scale=(0.30, 0.70)`` (v2)    483 / **1,419** / 4,730       6.8x
===================================== ============================= =============

A local view under the submitted recipe is a median **24x24 px** fragment of a
single seed inflated to 65,536 output pixels -- 0.9 % real content -- and **8 of
the 10 cross-view terms** in Eq. 1 are anchored on one. That is the single
largest mismatch between the objective and this data, and it is what the ``v2``
augmentation policy in ``conf/data/seed_crops_v2.yaml`` exists to fix.

Three mechanisms carry the fix, and each is separately switchable so the arms
stay single-factor:

``crop_ratio``
    The aspect-ratio range ``RandomResizedCrop`` samples. Torchvision's default
    ``(3/4, 4/3)`` cannot fit a high-area crop inside a *non-square* source, and
    only **3.4 %** of these crops are square (aspect p5/p95 = 0.52 / 1.98). After
    10 failed attempts ``get_params`` falls back to a **deterministic centre
    crop**, so raising the scale floor silently trades randomness for content:
    measured fallback rates are 3.5 % at ``scale=(0.40, 1.00)`` and **22.0 %** at
    ``(0.70, 1.00)`` with the default ratio, against **10.2 %** at
    ``crop_ratio=(0.5, 2.0)``. Widening the ratio is therefore strictly better
    than not widening it -- more native content *and* more diversity.

``min_native_pixels``
    A floor on the number of source pixels a view is allowed to be built from.
    The scale range is a *fraction*, so a fixed range means a 21x22 crop and an
    881x413 crop are shredded by the same factor. This raises the effective
    lower scale bound per image, ``max(scale_lo, min_native_pixels / area)``,
    and never lowers it -- so it can only make views less destructive. ``0``
    disables it, which is the pre-v2 behaviour exactly.

``rotation90_prob`` / ``vertical_flip_prob``
    Seeds photographed on a tray have no canonical orientation, so the dihedral
    group of 8 is label-preserving here in a way it is not for ImageNet. The
    90-degree rotations go through ``PIL.Image.transpose``, which is a pixel
    permutation: **no interpolation, no resampling, no black corners**. This buys
    view diversity back after the crop ranges narrow, which is what keeps the
    narrower ranges from turning into pretext-task memorisation on 81 scenes.

Colour is signal here, not only nuisance
----------------------------------------

The standard SSL argument for aggressive colour jitter is that colour histograms
are a shortcut. On this dataset mean RGB alone scores **0.3169** on the 27-way
task and explains 78.5 % of the between-sub-variety variance, and only ~26.6 %
of the within-class colour variance is photograph-specific -- so most of the
colour cue *transfers* (``STAGE1_CHANGES.md`` C5).

The v2 policy therefore splits the jitter by what it attacks rather than turning
it down uniformly:

* **brightness and contrast** are illumination, which *is* photograph-specific
  nuisance. Left at the reference +-0.4.
* **saturation and hue** are pigmentation, which is class signal. Cut to +-0.1
  and +-0.02; the submitted +-0.1 hue is +-36 degrees of the hue circle.
* **grayscale** deletes the cue outright, and **solarization** inverts it. Both
  go to 0 in v2 and both remain config keys, so ``wo_colour_preservation`` is a
  one-line arm rather than a fork.

Note what this is *not* claiming. The audit refuted the obvious version of this
argument: mean RGB is **more** linearly decodable after the shipped DINO run than
before (pooled R^2 0.864 -> 0.908), so the jitter did not destroy the cue. The
change is about where capacity goes, and it is an arm, not a certainty.

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

Where the per-view work happens
-------------------------------

Two flags move work off the dataloader workers without changing the arithmetic,
because at batch 16 x 6 views the CPU pipeline is what starves the GPU:

``output_uint8``
    Stop the per-view pipeline at ``PILToTensor`` and let the trainer do
    ``/255 -> normalize`` on the GPU. The tensors that get collated, pinned and
    copied over PCIe are then 1 byte per channel instead of 4 -- a 4x cut in
    every one of those three costs, for arithmetic that is identical up to
    float association.

``defer_local_upsample``
    Emit local crops at ``local_crop_size`` and let the trainer resize them to
    ``image_size`` on the GPU. A 101 px crop is **6.4x fewer pixels** than the
    256 px one it becomes, so this is the larger of the two savings.

The ordering is load-bearing and is preserved exactly. The CPU recipe is
``crop -> jitter -> blur -> ToTensor -> Normalize -> Resize``: the upsample
happens *after* normalisation, on unclamped floats. The GPU path therefore also
normalises first and interpolates second. Doing it the other way -- resizing
uint8 and normalising after -- would clamp the bicubic overshoot into [0, 255]
and quantise it, which is a different transform. That is why ``output_uint8``
forces ``defer_local_upsample``: there is no correct way to apply the existing
resize to a uint8 tensor.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps

#: Aspect-ratio range ``RandomResizedCrop`` samples, as a config default.
#:
#: Torchvision's own default. Kept as a named constant because the v2 policy
#: widens it and the *reason* is a measured property of this dataset rather than
#: a preference -- see the module docstring's ``crop_ratio`` entry.
TORCHVISION_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)

#: The four ways a source image can be mapped into a square view, exposed so a
#: run records which one it used rather than leaving it to be inferred from the
#: transform's repr.
DIHEDRAL_TRANSPOSES = (
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


class RandomRotation90:
    """Rotate by a random multiple of 90 degrees, losslessly.

    ``T.RandomRotation`` resamples: it interpolates every pixel and leaves black
    corners wherever the rotated rectangle does not cover the frame. On images
    whose *entire* information content is a median 52x51 native pixels already
    upsampled 5x, that is a destructive operation to apply for the sake of an
    augmentation.

    ``PIL.Image.transpose`` with a ``ROTATE_*`` constant is a pure index
    permutation instead -- every output pixel is exactly some input pixel, no
    interpolation, no corners, and no cost beyond the copy. Combined with the
    horizontal and vertical flips it generates the dihedral group of order 8.

    This is safe *here* specifically because seeds photographed on a tray have no
    canonical orientation, so all eight elements are label-preserving. It is not
    a general-purpose default and is off unless a config asks for it.

    Args:
        probability: Chance of applying a rotation at all. When it fires, one of
            the three non-identity rotations is chosen uniformly, so the overall
            distribution over the four rotations is
            ``(1 - p, p/3, p/3, p/3)``.
    """

    def __init__(self, probability: float = 0.0):
        self.probability = float(probability)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.probability <= 0.0 or random.random() > self.probability:
            return image
        return image.transpose(random.choice(DIHEDRAL_TRANSPOSES))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.probability})"


class NativeFloorRandomResizedCrop(T.RandomResizedCrop):
    """``RandomResizedCrop`` with a floor on the *native pixels* behind a view.

    ``RandomResizedCrop``'s ``scale`` is a **fraction of the source area**, which
    is the right parameterisation when every source is the same size and wrong
    when they span 21x22 to 881x413 as they do here. A fixed ``scale=(0.30,
    0.70)`` takes 795-1,856 px from the median 52x51 crop and 12,600-29,400 px
    from a large one: the same nominal augmentation strength, two entirely
    different amounts of destruction.

    This raises the lower bound per image to
    ``max(scale_lo, min_native_pixels / area)``, clipped so it can never exceed
    the upper bound. It therefore **only ever makes views less destructive**, and
    with ``min_native_pixels=0`` it is ``RandomResizedCrop`` exactly -- which is
    what makes the two a single-factor comparison.

    The floor is not a guarantee. ``get_params`` still samples an aspect ratio
    and can fall back to a centre crop, and the target box is clipped to the
    source, so the realised pixel count can land below the floor. It is a shift
    of the distribution, and ``view_geometry_report`` measures the realised one.
    """

    def __init__(self, *args: Any, min_native_pixels: float = 0.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.min_native_pixels = max(float(min_native_pixels), 0.0)

    def _effective_scale(self, width: int, height: int) -> list[float]:
        low, high = float(self.scale[0]), float(self.scale[1])
        if self.min_native_pixels <= 0.0:
            return [low, high]
        area = max(float(width) * float(height), 1.0)
        # `high` is the ceiling, not a target: a source smaller than the floor
        # simply gets the largest crop the range allows.
        return [min(max(low, self.min_native_pixels / area), high), high]

    def forward(self, img):  # type: ignore[override]
        # torchvision's own forward, with the scale range recomputed per image.
        # Re-implemented rather than wrapped because `get_params` is a static
        # method that takes the range as an argument, so there is no hook.
        from torchvision.transforms import functional as VF

        width, height = VF.get_image_size(img)
        top, left, crop_h, crop_w = self.get_params(
            img, self._effective_scale(width, height), list(self.ratio)
        )
        return VF.resized_crop(
            img, top, left, crop_h, crop_w, self.size, self.interpolation, antialias=True
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(size={self.size}, scale={tuple(self.scale)}, "
            f"ratio={tuple(round(value, 4) for value in self.ratio)}, "
            f"min_native_pixels={self.min_native_pixels:g})"
        )


def build_resized_crop(
    size: int,
    scale: Sequence[float],
    ratio: Sequence[float],
    interpolation: Any,
    min_native_pixels: float = 0.0,
) -> T.RandomResizedCrop:
    """The crop operator for one view family.

    Returns a plain ``T.RandomResizedCrop`` when there is no native-pixel floor,
    so a run that does not use the floor is byte-identical to one built before
    :class:`NativeFloorRandomResizedCrop` existed.
    """
    if float(min_native_pixels) > 0.0:
        return NativeFloorRandomResizedCrop(
            size,
            scale=tuple(scale),
            ratio=tuple(ratio),
            interpolation=interpolation,
            min_native_pixels=float(min_native_pixels),
        )
    return T.RandomResizedCrop(
        size, scale=tuple(scale), ratio=tuple(ratio), interpolation=interpolation
    )


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
        output_uint8: Emit ``uint8`` CHW tensors and leave ``/255 -> normalize``
            to the GPU. Forces ``defer_local_upsample``; see the module
            docstring for why.
        defer_local_upsample: Emit local crops at ``local_crop_size`` and leave
            the upsample to ``image_size`` to the GPU. Only meaningful with
            ``resize_local_to_global``.
        return_original: Also build the un-augmented full-frame view. Stage 1
            never reads it -- the trainer discards it and the attention overlay
            uses ``views[0]`` -- so producing it costs a resize, a collate and a
            256x256x3 float per sample per epoch for nothing. Left ``True`` by
            default so the ``(original, crops)`` contract holds for callers that
            do want it; the pretraining config turns it off.
    """

    def __init__(
        self,
        image_size: int,
        local_crop_size: int,
        global_crops_scale: Sequence[float] = (0.4, 1.0),
        local_crops_scale: Sequence[float] = (0.05, 0.4),
        local_crops_number: int = 4,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,
        rotation90_prob: float = 0.0,
        color_jitter_prob: float = 0.8,
        color_jitter_brightness: float = 0.4,
        color_jitter_contrast: float = 0.4,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
        grayscale_prob: float = 0.2,
        global_blur_prob_1: float = 1.0,
        global_blur_prob_2: float = 0.1,
        local_blur_prob: float = 0.5,
        blur_radius_min: float = 0.1,
        blur_radius_max: float = 2.0,
        solarization_prob: float = 0.2,
        crop_ratio: Sequence[float] = TORCHVISION_CROP_RATIO,
        min_native_pixels: float = 0.0,
        resize_local_to_global: bool = True,
        match_view_lowpass: bool = False,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
        output_uint8: bool = False,
        defer_local_upsample: bool = False,
        return_original: bool = True,
    ):
        if local_crops_number < 0:
            raise ValueError(f"local_crops_number must be >= 0, got {local_crops_number}")
        crop_ratio = tuple(float(value) for value in crop_ratio)
        if len(crop_ratio) != 2 or not 0 < crop_ratio[0] <= crop_ratio[1]:
            raise ValueError(
                f"crop_ratio must be an ordered pair of positive aspect ratios, got {crop_ratio!r}"
            )
        if float(min_native_pixels) < 0:
            raise ValueError(f"min_native_pixels must be >= 0, got {min_native_pixels}")

        # A uint8 view cannot carry the post-normalisation resize, so asking for
        # one implies deferring that resize to the GPU. Recorded rather than
        # silently assumed: `defer_local_upsample` is public and the trainer logs
        # it alongside the view geometry.
        self.output_uint8 = bool(output_uint8)
        self.defer_local_upsample = bool(defer_local_upsample) or self.output_uint8
        self.return_original = bool(return_original)
        self.image_size = int(image_size)
        self.local_crop_size = int(local_crop_size)
        self.normalize_mean = tuple(float(value) for value in normalize_mean)
        self.normalize_std = tuple(float(value) for value in normalize_std)

        # Kept as attributes because `describe()` reports them into
        # `summary.json` and the geometry report re-derives the realised native
        # pixel distribution from exactly these numbers. Reading the policy back
        # off a `T.Compose` repr is not something a downstream script should have
        # to do.
        self.global_crops_scale = tuple(float(value) for value in global_crops_scale)
        self.local_crops_scale = tuple(float(value) for value in local_crops_scale)
        self.crop_ratio = crop_ratio
        self.min_native_pixels = float(min_native_pixels)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.vertical_flip_prob = float(vertical_flip_prob)
        self.rotation90_prob = float(rotation90_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter_brightness = float(color_jitter_brightness)
        self.color_jitter_contrast = float(color_jitter_contrast)
        self.color_jitter_saturation = float(color_jitter_saturation)
        self.color_jitter_hue = float(color_jitter_hue)
        self.grayscale_prob = float(grayscale_prob)
        self.global_blur_prob_1 = float(global_blur_prob_1)
        self.global_blur_prob_2 = float(global_blur_prob_2)
        self.local_blur_prob = float(local_blur_prob)
        self.blur_radius_min = float(blur_radius_min)
        self.blur_radius_max = float(blur_radius_max)
        self.solarization_prob = float(solarization_prob)
        self.match_view_lowpass = bool(match_view_lowpass)
        self.local_crops_number = int(local_crops_number)

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
        # Geometry first, then photometry. The dihedral steps are lossless index
        # permutations (see RandomRotation90), so their position in the chain is
        # a matter of cost rather than of arithmetic -- they run before the
        # jitter because jittering fewer distinct pixel arrangements is the same
        # work and rotating a jittered image is the same result.
        geometry: list[Any] = [T.RandomHorizontalFlip(p=horizontal_flip_prob)]
        if vertical_flip_prob > 0:
            geometry.append(T.RandomVerticalFlip(p=vertical_flip_prob))
        if rotation90_prob > 0:
            geometry.append(RandomRotation90(rotation90_prob))

        photometry: list[Any] = [
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
            )
        ]
        # Not allocated at zero: `RandomGrayscale(p=0)` is a no-op that still
        # costs a `random()` draw and a branch per view per epoch, and its
        # presence in the transform's repr would suggest the run used it.
        if grayscale_prob > 0:
            photometry.append(T.RandomGrayscale(p=grayscale_prob))
        flip_and_color_jitter = T.Compose([*geometry, *photometry])
        # `PILToTensor` is the uint8 half of `ToTensor`: same HWC -> CHW
        # permutation, no division, no float allocation. The trainer applies the
        # remaining `/255 -> (x - mean) / std` on the GPU.
        normalize = (
            T.PILToTensor()
            if self.output_uint8
            else T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.normalize_mean, self.normalize_std),
                ]
            )
        )

        # Un-augmented view, kept for visualisation and attention overlays.
        self.original_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=interpolation),
                T.ToTensor(),
            ]
        )
        def blur(probability: float) -> GaussianBlur:
            return GaussianBlur(
                probability, radius_min=self.blur_radius_min, radius_max=self.blur_radius_max
            )

        def global_crop() -> T.RandomResizedCrop:
            return build_resized_crop(
                image_size,
                self.global_crops_scale,
                self.crop_ratio,
                interpolation,
                min_native_pixels=self.min_native_pixels,
            )

        self.global_transform_1 = T.Compose(
            [
                global_crop(),
                flip_and_color_jitter,
                blur(global_blur_prob_1),
                *lowpass,
                normalize,
            ]
        )
        global_2_steps: list[Any] = [
            global_crop(),
            flip_and_color_jitter,
            blur(global_blur_prob_2),
        ]
        # Solarization inverts every pixel above the threshold, which on a
        # pigmentation-discriminative task destroys the cue rather than making
        # the model invariant to a nuisance. Not allocated at zero, for the same
        # reason grayscale is not.
        if solarization_prob > 0:
            global_2_steps.append(Solarization(solarization_prob))
        self.global_transform_2 = T.Compose([*global_2_steps, *lowpass, normalize])

        local_steps: list[Any] = [
            build_resized_crop(
                local_crop_size,
                self.local_crops_scale,
                self.crop_ratio,
                interpolation,
                min_native_pixels=self.min_native_pixels,
            ),
            flip_and_color_jitter,
            blur(local_blur_prob),
            normalize,
        ]
        # SwinV2 needs a fixed input resolution; the local/global distinction is
        # carried by the crop scale, not by the final tensor size. The upsample
        # happens here, or on the GPU when it is deferred -- in both cases
        # *after* normalisation, so the two paths are the same function.
        self.resize_local_to_global = bool(resize_local_to_global)
        self.upsample_locals_on_device = self.resize_local_to_global and self.defer_local_upsample
        if self.resize_local_to_global and not self.defer_local_upsample:
            local_steps.append(T.Resize((image_size, image_size), interpolation=interpolation))

        self.local_transform = T.Compose(local_steps)

    def describe(self) -> dict[str, Any]:
        """The resolved augmentation policy, as plain JSON-safe scalars.

        The trainer writes this into ``summary.json`` and the CSV sink, so an arm
        is machine-distinguishable from the file rather than from its directory
        name. Two arms whose configs differ only in ``crop_ratio`` produce
        different dictionaries here; two arms that differ only in which YAML file
        set the same values produce identical ones, which is the correct
        behaviour for a provenance record.
        """
        return {
            "image_size": self.image_size,
            "local_crop_size": self.local_crop_size,
            "local_crops_number": self.local_crops_number,
            "num_crops": self.num_crops,
            "global_crops_scale": list(self.global_crops_scale),
            "local_crops_scale": list(self.local_crops_scale),
            "crop_ratio": list(self.crop_ratio),
            "min_native_pixels": self.min_native_pixels,
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "vertical_flip_prob": self.vertical_flip_prob,
            "rotation90_prob": self.rotation90_prob,
            "color_jitter_prob": self.color_jitter_prob,
            "color_jitter_brightness": self.color_jitter_brightness,
            "color_jitter_contrast": self.color_jitter_contrast,
            "color_jitter_saturation": self.color_jitter_saturation,
            "color_jitter_hue": self.color_jitter_hue,
            "grayscale_prob": self.grayscale_prob,
            "global_blur_prob_1": self.global_blur_prob_1,
            "global_blur_prob_2": self.global_blur_prob_2,
            "local_blur_prob": self.local_blur_prob,
            "blur_radius_min": self.blur_radius_min,
            "blur_radius_max": self.blur_radius_max,
            "solarization_prob": self.solarization_prob,
            "match_view_lowpass": self.match_view_lowpass,
            "resize_local_to_global": self.resize_local_to_global,
            "output_uint8": self.output_uint8,
            "defer_local_upsample": self.defer_local_upsample,
            "upsample_locals_on_device": self.upsample_locals_on_device,
            "view_sizes": list(self.view_sizes),
            "normalize_mean": list(self.normalize_mean),
            "normalize_std": list(self.normalize_std),
            # The dihedral group has 8 elements only when both flips and the
            # rotations are live; recorded as a number so a plot axis can use it.
            "dihedral_group_order": (
                8
                if self.rotation90_prob > 0 and self.vertical_flip_prob > 0
                else 4
                if self.rotation90_prob > 0 or self.vertical_flip_prob > 0
                else 2
            ),
        }

    @property
    def num_crops(self) -> int:
        """Total views per image: 2 global + ``local_crops_number`` local."""
        return 2 + self.local_crops_number

    @property
    def local_view_size(self) -> int:
        """Spatial size the local views are emitted at.

        Equal to ``image_size`` unless the upsample is deferred to the GPU, in
        which case the tensors that cross the dataloader boundary are
        ``local_crop_size`` and the trainer finishes the job.
        """
        if self.resize_local_to_global and not self.defer_local_upsample:
            return self.image_size
        return self.local_crop_size

    @property
    def view_sizes(self) -> list[int]:
        """Emitted spatial size of each view, in ``view_ids`` order."""
        return [self.image_size, self.image_size] + [self.local_view_size] * self.local_crops_number

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

    def __call__(
        self,
        image: Image.Image,
        partner_images: Sequence[Image.Image] | None = None,
    ):
        """Return ``(original_tensor, [global_1, global_2, *locals])``.

        ``original_tensor`` is ``None`` when ``return_original=False``; the
        multi-crop collate drops it in that case rather than trying to batch it.

        ``partner_images`` replaces the **trailing** local views with local crops
        of those images instead of the anchor -- the provenance-derived positives
        of ``STAGE1_CHANGES.md`` F1, where the partners are other crops of the
        same source photograph. The view *count* is unchanged, so the loss's
        cross-view pairing, ``view_ids`` and every shape downstream are untouched;
        only what view 2..V depict changes. Supplying more partners than there are
        local views is an error rather than a silent truncation, because the
        difference between "two of four" and "all four" is the arm.
        """
        image = image.convert("RGB")
        partners = list(partner_images or ())
        if len(partners) > self.local_crops_number:
            raise ValueError(
                f"{len(partners)} partner images were supplied for {self.local_crops_number} "
                "local views; there is nowhere to put the extras."
            )
        crops = [self.global_transform_1(image), self.global_transform_2(image)]
        anchors = [image] * (self.local_crops_number - len(partners)) + [
            partner.convert("RGB") for partner in partners
        ]
        crops.extend(self.local_transform(source) for source in anchors)
        original = self.original_transform(image) if self.return_original else None
        return original, crops


def get_dino_transforms(
    image_size: int,
    local_crop_size: int,
    augmentation_cfg: Any | None = None,
    **overrides: Any,
):
    """Build :class:`DataAugmentationDINO` from a ``data.augmentation`` config node.

    ``overrides`` take precedence over the config node, which is how the trainer
    switches off ``return_original`` without that being an augmentation *policy*
    recorded in ``conf/data/``.
    """
    params = dict(augmentation_cfg or {})
    # `local_crop_size` lives on the data node, not inside `augmentation`.
    params.pop("local_crop_size", None)
    # `same_photo_local_views` is an *augmentation policy* -- which is why it
    # lives in this node -- but it is consumed by the dataset, which owns the
    # source-photograph index needed to draw a partner. The transform only has to
    # accept partner images when it is handed some, so the key is dropped here
    # rather than becoming a constructor argument that would have nothing to do.
    params.pop("same_photo_local_views", None)
    params.update({key: value for key, value in overrides.items() if value is not None})
    return DataAugmentationDINO(image_size=image_size, local_crop_size=local_crop_size, **params)


def _percentiles(values: Any, points: Sequence[float] = (5, 25, 50, 75, 95)) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {f"p{int(point)}": float("nan") for point in points}
    quantiles = np.percentile(array, list(points))
    return {f"p{int(point)}": float(value) for point, value in zip(points, quantiles)}


def measure_view_geometry(
    source_sizes: Sequence[tuple[int, int]],
    scale: Sequence[float],
    ratio: Sequence[float],
    output_size: int,
    min_native_pixels: float = 0.0,
    samples: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    """How many **native source pixels** one view family is actually built from.

    This calls torchvision's own :meth:`RandomResizedCrop.get_params` against the
    real distribution of source sizes rather than modelling it, because the two
    disagree in a way that matters here. ``get_params`` retries the
    (area, aspect) draw ten times and then falls back to a **deterministic centre
    crop**; on a corpus that is 96.6 % non-square, raising the scale floor pushes
    the sampler into that fallback and silently trades augmentation diversity for
    content. ``deterministic_fallback_rate`` is that quantity, and it is the
    reason ``crop_ratio`` exists as a config key.

    Returns native-pixel percentiles, the implied upsample factor to
    ``output_size``, the fallback rate, and the share of views falling below the
    ``min_native_pixels`` floor. Every key is a plain float, so the trainer
    writes the whole dictionary into a CSV row unmodified.

    Args:
        source_sizes: ``(width, height)`` per source image, in pixels.
        scale / ratio: The area and aspect ranges the crop samples.
        output_size: Spatial size the crop is resized to.
        min_native_pixels: The floor :class:`NativeFloorRandomResizedCrop`
            applies, or ``0`` for a plain ``RandomResizedCrop``.
        samples: Monte-Carlo draws. 20k lands the median within ~1 %.
        seed: Draw seed, so the report is reproducible.
    """
    import numpy as np
    import torch
    from torchvision.transforms import RandomResizedCrop

    sizes = np.asarray(list(source_sizes), dtype=np.int64).reshape(-1, 2)
    if sizes.size == 0:
        raise ValueError("measure_view_geometry needs at least one source size")

    rng = np.random.default_rng(int(seed))
    chosen = rng.integers(0, sizes.shape[0], int(samples))
    low, high = float(scale[0]), float(scale[1])
    ratio_pair = [float(ratio[0]), float(ratio[1])]

    native: list[float] = []
    fallbacks = 0
    # `get_params` draws from the global torch RNG. Forking it keeps a geometry
    # report -- which the trainer runs at startup, before the first batch -- from
    # advancing the stream the augmentation pipeline itself samples from.
    with torch.random.fork_rng(devices=[]):
        torch.random.manual_seed(int(seed))
        for index in chosen:
            width, height = int(sizes[index, 0]), int(sizes[index, 1])
            area = float(width * height)
            effective_low = (
                min(max(low, float(min_native_pixels) / max(area, 1.0)), high)
                if float(min_native_pixels) > 0
                else low
            )
            dummy = torch.empty(3, height, width)
            top, left, crop_h, crop_w = RandomResizedCrop.get_params(
                dummy, [effective_low, high], ratio_pair
            )
            native.append(float(crop_h * crop_w))
            # Reproduce torchvision's fallback box and compare. There is no flag
            # on the return value, so the box itself is the only evidence -- and
            # a random draw that happens to land exactly on the fallback box is
            # indistinguishable from the fallback, which biases this upward by a
            # negligible amount and never downward.
            in_ratio = width / height
            if in_ratio < ratio_pair[0]:
                fallback_w, fallback_h = width, int(round(width / ratio_pair[0]))
            elif in_ratio > ratio_pair[1]:
                fallback_h, fallback_w = height, int(round(height * ratio_pair[1]))
            else:
                fallback_w, fallback_h = width, height
            if (top, left, crop_h, crop_w) == (
                (height - fallback_h) // 2,
                (width - fallback_w) // 2,
                fallback_h,
                fallback_w,
            ):
                fallbacks += 1

    native_array = np.asarray(native, dtype=np.float64)
    percentiles = _percentiles(native_array)
    side = np.sqrt(np.maximum(native_array, 1.0))
    report = {
        "output_size": float(output_size),
        "scale_low": low,
        "scale_high": high,
        "ratio_low": ratio_pair[0],
        "ratio_high": ratio_pair[1],
        "min_native_pixels": float(min_native_pixels),
        "samples": float(native_array.size),
        "native_pixels_mean": float(native_array.mean()),
        **{f"native_pixels_{key}": value for key, value in percentiles.items()},
        "native_side_median": float(np.median(side)),
        "upsample_factor_median": float(np.median(float(output_size) / side)),
        "upsample_factor_p95": float(np.percentile(float(output_size) / side, 95)),
        # The share of the 65,536 output pixels of a 256 px view that correspond
        # to a real sensor measurement. The headline number of STAGE1_CHANGES 0.5.
        "real_content_fraction_median": float(
            np.median(native_array) / (float(output_size) ** 2)
        ),
        "deterministic_fallback_rate": float(fallbacks) / float(native_array.size),
        "below_floor_fraction": (
            float((native_array < float(min_native_pixels)).mean())
            if float(min_native_pixels) > 0
            else 0.0
        ),
    }
    return report


def view_geometry_report(
    transform: DataAugmentationDINO,
    source_sizes: Sequence[tuple[int, int]],
    samples: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """:func:`measure_view_geometry` for both view families of one transform.

    The one number to read first is ``local.native_pixels_p50``. Under the
    submitted recipe it is ~598 -- a 24x24 px fragment of one seed carrying 0.9 %
    of the 65,536 pixels the encoder is handed -- and **8 of the 10 cross-view
    terms** in Eq. 1 are anchored on such a view.
    """
    global_report = measure_view_geometry(
        source_sizes,
        transform.global_crops_scale,
        transform.crop_ratio,
        transform.image_size,
        min_native_pixels=transform.min_native_pixels,
        samples=samples,
        seed=seed,
    )
    local_report = measure_view_geometry(
        source_sizes,
        transform.local_crops_scale,
        transform.crop_ratio,
        # The local crop is emitted at `local_crop_size` and upsampled to
        # `image_size` afterwards, so the factor that matters end to end is
        # against the size the encoder sees, not the intermediate.
        transform.image_size,
        min_native_pixels=transform.min_native_pixels,
        samples=samples,
        seed=seed + 1,
    )
    import numpy as np

    sizes = np.asarray(list(source_sizes), dtype=np.float64).reshape(-1, 2)
    areas = sizes[:, 0] * sizes[:, 1]
    return {
        "policy": transform.describe(),
        "global": global_report,
        "local": local_report,
        "source": {
            "count": float(sizes.shape[0]),
            "width_median": float(np.median(sizes[:, 0])),
            "height_median": float(np.median(sizes[:, 1])),
            **{f"area_{key}": value for key, value in _percentiles(areas).items()},
            "square_fraction": float((sizes[:, 0] == sizes[:, 1]).mean()),
            "both_sides_under_64_fraction": float(
                ((sizes[:, 0] < 64) & (sizes[:, 1] < 64)).mean()
            ),
            "aspect_ratio_p5": float(np.percentile(sizes[:, 0] / sizes[:, 1], 5)),
            "aspect_ratio_p95": float(np.percentile(sizes[:, 0] / sizes[:, 1], 95)),
        },
        # The share of Eq. 1's cross-view terms anchored on a local view. With
        # 2 teacher globals and V student views the pairs are 2V - 2, of which
        # 2 * local_crops_number involve a local student view.
        "local_anchored_loss_term_fraction": (
            (2.0 * transform.local_crops_number) / max(2.0 * transform.num_crops - 2.0, 1.0)
        ),
    }


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
