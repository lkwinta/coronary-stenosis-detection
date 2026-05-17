from skimage import morphology

import numpy as np


def clean_mask(
    mask: np.ndarray,
    closing_radius: float = 2,
    max_hole_size: int = 50,
    min_object_size: int = 50,
) -> np.ndarray:
    binary = mask.astype(bool)

    if closing_radius > 0:
        binary = morphology.closing(binary, morphology.disk(closing_radius))

    binary = morphology.remove_small_holes(binary, area_threshold=max_hole_size)
    binary = morphology.remove_small_objects(binary, min_size=min_object_size)

    return binary.astype(np.uint8)
