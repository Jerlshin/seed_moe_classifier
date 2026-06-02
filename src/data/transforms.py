import torchvision.transforms as T
from lightly.transforms import DINOTransform

def get_dino_transforms(image_size: int, local_crop_size: int):
    """
    Creates the DINO multi-crop transformations as specified in the notebook.
    Generates 2 global views and multiple local views.
    """
    transform = DINOTransform(
        global_crop_size=image_size,
        local_crop_size=local_crop_size,
        n_local_views=6
    )
    return transform
